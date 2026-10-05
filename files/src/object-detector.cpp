/*
 * ============================================================
 * YOLO26 MERGED PIPELINE
 * ============================================================
 *
 * Combines:
 *   - File 1: snapshot capture + JSON detection events
 *   - File 2: live annotated RTSP re-stream
 *
 * RTSP INPUT:
 *   Read automatically from:
 *   /var/lib/camera-discovery/cameras.json
 *
 * The rest of the pipeline behavior is unchanged.
 *
 * Pipeline:
 *
 *   nvurisrcbin -> streammux -> nvinfer -> nvdsosd -> tee
 *        |                                             |-- queue -> fakesink
 *        |                                             |-- queue -> nvvideoconvert ->
 *        |                                             |            capsfilter -> nvjpegenc ->
 *        |                                             |            appsink
 *        |                                             |-- queue -> nvrtspoutsinkbin
 *
 * ============================================================
 */

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <glib.h>
#include <glib-unix.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <csignal>
#include <string>
#include <map>
#include <mutex>
#include <vector>
#include <cctype>

#include <nlohmann/json.hpp>

#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

using json = nlohmann::json;


/*
 * ============================================================
 * Configuration
 * ============================================================
 */

#define SNAPSHOT_DIR              "/root/video_recorder/snapshots"
#define EVENT_DIR                 "/root/video_recorder/events/pending"
#define DEEPSTREAM_CONFIG         "/opt/radian/ai/config/config_infer_primary_yolo26.txt"
#define SNAPSHOT_COOLDOWN_SECONDS 5
#define RTSP_OUTPUT_URL_HINT      "rtsp://192.168.1.155:8554/ai"

/* Camera discovery output */
#define CAMERA_JSON_FILE          "/var/lib/camera-discovery/cameras.json"


/*
 * ============================================================
 * Shared state — touched from multiple threads, hence mutex.
 * ============================================================
 */

struct SnapshotData
{
    std::vector<unsigned char> jpeg_data;
    bool valid = false;
};

static SnapshotData latest_snapshot;
static std::mutex snapshot_mutex;
static std::map<std::string, time_t> last_snapshot_time;

static GMainLoop *main_loop = nullptr;


/*
 * ============================================================
 * Read RTSP URL from camera-discovery JSON
 * ============================================================
 *
 * Expected JSON:
 *
 * {
 *     "version": 1,
 *     "updated_at": "...",
 *     "cameras": [
 *         {
 *             "id": "camera_001",
 *             "mac": "...",
 *             "ip": "10.10.10.156",
 *             "status": "online",
 *             "rtsp": {
 *                 "valid": true,
 *                 "url": "rtsp://..."
 *             }
 *         }
 *     ]
 * }
 *
 * The first camera that is:
 *
 *     status = online
 *     rtsp.valid = true
 *
 * will be selected.
 *
 * ============================================================
 */

static std::string get_rtsp_url_from_json()
{
    std::ifstream file(CAMERA_JSON_FILE);

    if (!file.is_open())
    {
        std::cerr << "ERROR: Cannot open camera JSON file: "
                  << CAMERA_JSON_FILE << std::endl;

        return "";
    }

    try
    {
        json data;
        file >> data;

        if (!data.contains("cameras") ||
            !data["cameras"].is_array())
        {
            std::cerr << "ERROR: Invalid camera JSON: "
                      << "'cameras' array not found."
                      << std::endl;

            return "";
        }

        for (const auto &camera : data["cameras"])
        {
            /*
             * Check camera status
             */
            if (!camera.contains("status") ||
                camera["status"] != "online")
            {
                continue;
            }

            /*
             * Check RTSP section
             */
            if (!camera.contains("rtsp") ||
                !camera["rtsp"].is_object())
            {
                continue;
            }

            /*
             * Check RTSP validity
             */
            if (!camera["rtsp"].contains("valid") ||
                !camera["rtsp"]["valid"].get<bool>())
            {
                continue;
            }

            /*
             * Check RTSP URL
             */
            if (!camera["rtsp"].contains("url") ||
                !camera["rtsp"]["url"].is_string())
            {
                continue;
            }

            std::string camera_id =
                camera.value("id", "unknown");

            std::string rtsp_url =
                camera["rtsp"]["url"].get<std::string>();

            if (rtsp_url.empty())
            {
                continue;
            }

            std::cout << "========================================"
                      << std::endl;

            std::cout << "Camera ID : "
                      << camera_id
                      << std::endl;

            std::cout << "Camera IP : "
                      << camera.value("ip", "unknown")
                      << std::endl;

            std::cout << "RTSP URL  : "
                      << rtsp_url
                      << std::endl;

            std::cout << "========================================"
                      << std::endl;

            return rtsp_url;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "ERROR: Failed to parse "
                  << CAMERA_JSON_FILE
                  << ": "
                  << e.what()
                  << std::endl;

        return "";
    }

    std::cerr << "ERROR: No valid online camera found in "
              << CAMERA_JSON_FILE
              << std::endl;

    return "";
}


/*
 * ============================================================
 * Small helpers
 * ============================================================
 */

static std::string get_timestamp()
{
    std::time_t now = std::time(nullptr);
    std::tm tm_now;
    gmtime_r(&now, &tm_now);

    std::ostringstream oss;
    oss << std::put_time(&tm_now, "%Y-%m-%dT%H:%M:%S");

    return oss.str();
}


static bool save_jpeg_data(
    const std::vector<unsigned char> &data,
    const std::string &filename)
{
    if (data.empty())
    {
        g_printerr("JPEG data is empty\n");
        return false;
    }

    std::ofstream file(filename, std::ios::binary);

    if (!file)
    {
        g_printerr(
            "Failed to open snapshot file: %s\n",
            filename.c_str());

        return false;
    }

    file.write(
        reinterpret_cast<const char *>(data.data()),
        static_cast<std::streamsize>(data.size()));

    file.close();

    if (!file)
    {
        g_printerr(
            "Failed while writing snapshot: %s\n",
            filename.c_str());

        return false;
    }

    return true;
}


static std::string safe_filename_component(
    const std::string &value)
{
    std::string result;

    for (char c : value)
    {
        unsigned char uc =
            static_cast<unsigned char>(c);

        result +=
            (std::isalnum(uc) ||
             c == '_' ||
             c == '-') ? c : '_';
    }

    if (result.empty())
        result = "object";

    return result;
}


static std::string get_snapshot_category(
    const std::string &object_name)
{
    static const std::map<std::string, std::string>
        category_map = {

        {"bicycle", "vehicles"},
        {"car", "vehicles"},
        {"motorcycle", "vehicles"},
        {"bus", "vehicles"},
        {"train", "vehicles"},
        {"truck", "vehicles"},

        {"person", "people"},

        {"bird", "animals"},
        {"cat", "animals"},
        {"dog", "animals"},
        {"horse", "animals"},
        {"sheep", "animals"},
        {"cow", "animals"},
        {"elephant", "animals"},
        {"bear", "animals"},
        {"zebra", "animals"},
        {"giraffe", "animals"},

        {"cell phone", "electronics"},
        {"laptop", "electronics"},
        {"tv", "electronics"},
        {"keyboard", "electronics"},
        {"mouse", "electronics"},
        {"remote", "electronics"},
    };

    auto it = category_map.find(object_name);

    return (it != category_map.end())
        ? it->second
        : "other";
}


static std::string json_escape(
    const std::string &value)
{
    std::string result;

    for (char c : value)
    {
        switch (c)
        {
            case '\\':
                result += "\\\\";
                break;

            case '"':
                result += "\\\"";
                break;

            case '\n':
                result += "\\n";
                break;

            case '\r':
                result += "\\r";
                break;

            case '\t':
                result += "\\t";
                break;

            default:
                result += c;
                break;
        }
    }

    return result;
}


static bool save_detection_event(
    const std::string &object_name,
    float confidence,
    std::time_t detection_time,
    const std::string &timestamp,
    const std::string &snapshot_path)
{
    std::ostringstream filename;

    filename << EVENT_DIR << "/"
             << safe_filename_component(object_name)
             << "_"
             << timestamp
             << ".json";

    std::ofstream file(filename.str());

    if (!file)
    {
        g_printerr(
            "Failed to create detection event: %s\n",
            filename.str().c_str());

        return false;
    }

    file << "{\n"
         << "  \"object\": \""
         << json_escape(object_name)
         << "\",\n"

         << "  \"confidence\": "
         << std::fixed
         << std::setprecision(4)
         << confidence
         << ",\n"

         << "  \"timestamp\": \""
         << timestamp
         << "\",\n"

         << "  \"unix_timestamp\": "
         << static_cast<long long>(detection_time)
         << ",\n"

         << "  \"snapshot\": \""
         << json_escape(snapshot_path)
         << "\"\n"

         << "}\n";

    file.close();

    if (!file)
    {
        g_printerr(
            "Failed while writing detection event: %s\n",
            filename.str().c_str());

        return false;
    }

    g_print(
        "EVENT SAVED: %s\n",
        filename.str().c_str());

    return true;
}


static void save_snapshot_for_object(
    const std::string &object_name,
    float confidence)
{
    std::time_t now = std::time(nullptr);

    std::vector<unsigned char> jpeg_data;

    /*
     * Cooldown check + grab latest frame
     */
    {
        std::lock_guard<std::mutex>
            lock(snapshot_mutex);

        auto it =
            last_snapshot_time.find(object_name);

        if (it != last_snapshot_time.end() &&
            difftime(
                now,
                it->second)
                < SNAPSHOT_COOLDOWN_SECONDS)
        {
            return;
        }

        last_snapshot_time[object_name] = now;

        if (!latest_snapshot.valid ||
            latest_snapshot.jpeg_data.empty())
        {
            g_printerr(
                "No JPEG frame available for snapshot\n");

            return;
        }

        jpeg_data =
            latest_snapshot.jpeg_data;
    }

    std::string timestamp =
        get_timestamp();

    std::string category =
        get_snapshot_category(object_name);

    std::string category_dir =
        std::string(SNAPSHOT_DIR)
        + "/"
        + category;

    if (g_mkdir_with_parents(
            category_dir.c_str(),
            0755) != 0)
    {
        g_printerr(
            "Failed to create snapshot category directory: %s\n",
            category_dir.c_str());

        return;
    }

    std::ostringstream filename;

    filename << category_dir
             << "/"
             << safe_filename_component(object_name)
             << "_"
             << std::fixed
             << std::setprecision(2)
             << confidence
             << "_"
             << timestamp
             << ".jpg";

    if (save_jpeg_data(
            jpeg_data,
            filename.str()))
    {
        g_print(
            "SNAPSHOT SAVED: %s\n",
            filename.str().c_str());

        save_detection_event(
            object_name,
            confidence,
            now,
            timestamp,
            filename.str());
    }
}


/*
 * ============================================================
 * appsink callback
 * ============================================================
 */

static GstFlowReturn on_new_sample(
    GstAppSink *appsink,
    gpointer)
{
    GstSample *sample =
        gst_app_sink_pull_sample(appsink);

    if (!sample)
        return GST_FLOW_ERROR;

    GstBuffer *buffer =
        gst_sample_get_buffer(sample);

    if (!buffer)
    {
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }

    GstMapInfo map;

    if (gst_buffer_map(
            buffer,
            &map,
            GST_MAP_READ))
    {
        std::vector<unsigned char> jpeg(
            map.data,
            map.data + map.size);

        {
            std::lock_guard<std::mutex>
                lock(snapshot_mutex);

            latest_snapshot.jpeg_data =
                jpeg;

            latest_snapshot.valid =
                true;
        }

        gst_buffer_unmap(
            buffer,
            &map);
    }

    gst_sample_unref(sample);

    return GST_FLOW_OK;
}


/*
 * ============================================================
 * Detection probe
 * ============================================================
 */

static GstPadProbeReturn infer_src_probe(
    GstPad *,
    GstPadProbeInfo *info,
    gpointer)
{
    GstBuffer *buf =
        GST_PAD_PROBE_INFO_BUFFER(info);

    if (!buf)
        return GST_PAD_PROBE_OK;

    NvDsBatchMeta *batch_meta =
        gst_buffer_get_nvds_batch_meta(buf);

    if (!batch_meta)
        return GST_PAD_PROBE_OK;

    for (
        NvDsMetaList *l_frame =
            batch_meta->frame_meta_list;
        l_frame;
        l_frame = l_frame->next)
    {
        auto *frame_meta =
            static_cast<NvDsFrameMeta *>(
                l_frame->data);

        if (!frame_meta)
            continue;

        for (
            NvDsMetaList *l_obj =
                frame_meta->obj_meta_list;
            l_obj;
            l_obj = l_obj->next)
        {
            auto *obj_meta =
                static_cast<NvDsObjectMeta *>(
                    l_obj->data);

            if (!obj_meta)
                continue;

            std::string object_name =
                obj_meta->obj_label;

            float confidence =
                obj_meta->confidence;

            std::cout
                << "DETECTED: "
                << object_name
                << " confidence="
                << confidence
                << std::endl;

            save_snapshot_for_object(
                object_name,
                confidence);
        }
    }

    return GST_PAD_PROBE_OK;
}


/*
 * ============================================================
 * nvurisrcbin dynamic pad -> streammux sink_0
 * ============================================================
 */

static void on_pad_added(
    GstElement *,
    GstPad *new_pad,
    gpointer user_data)
{
    GstElement *streammux =
        GST_ELEMENT(user_data);

    GstCaps *caps =
        gst_pad_get_current_caps(new_pad);

    if (!caps)
        caps =
            gst_pad_query_caps(
                new_pad,
                nullptr);

    if (!caps)
        return;

    GstStructure *structure =
        gst_caps_get_structure(caps, 0);

    const gchar *name =
        gst_structure_get_name(structure);

    std::cout
        << "nvurisrcbin pad added: "
        << name
        << std::endl;

    if (!g_str_has_prefix(
            name,
            "video/"))
    {
        gst_caps_unref(caps);
        return;
    }

    GstPad *sink_pad =
        gst_element_request_pad_simple(
            streammux,
            "sink_0");

    if (!sink_pad)
    {
        std::cerr
            << "ERROR: Cannot get "
               "nvstreammux sink_0"
            << std::endl;

        gst_caps_unref(caps);
        return;
    }

    if (gst_pad_is_linked(sink_pad))
    {
        gst_object_unref(sink_pad);
        gst_caps_unref(caps);
        return;
    }

    GstPadLinkReturn ret =
        gst_pad_link(
            new_pad,
            sink_pad);

    if (ret == GST_PAD_LINK_OK)
    {
        std::cout
            << "Video linked to nvstreammux."
            << std::endl;
    }
    else
    {
        std::cerr
            << "ERROR: Failed to link video "
               "to nvstreammux. Return="
            << ret
            << std::endl;
    }

    gst_object_unref(sink_pad);
    gst_caps_unref(caps);
}


/*
 * ============================================================
 * Bus callback
 * ============================================================
 */

static gboolean bus_callback(
    GstBus *,
    GstMessage *message,
    gpointer user_data)
{
    GMainLoop *loop =
        static_cast<GMainLoop *>(user_data);

    switch (GST_MESSAGE_TYPE(message))
    {
        case GST_MESSAGE_ERROR:
        {
            GError *error = nullptr;
            gchar *debug = nullptr;

            gst_message_parse_error(
                message,
                &error,
                &debug);

            std::cerr
                << "\nGStreamer ERROR from "
                << GST_OBJECT_NAME(message->src)
                << ": "
                << error->message
                << std::endl;

            if (debug)
            {
                std::cerr
                    << "Debug: "
                    << debug
                    << std::endl;
            }

            g_error_free(error);
            g_free(debug);

            g_main_loop_quit(loop);

            break;
        }

        case GST_MESSAGE_WARNING:
        {
            GError *error = nullptr;
            gchar *debug = nullptr;

            gst_message_parse_warning(
                message,
                &error,
                &debug);

            std::cerr
                << "\nGStreamer WARNING from "
                << GST_OBJECT_NAME(message->src)
                << ": "
                << error->message
                << std::endl;

            if (debug)
            {
                std::cerr
                    << "Debug: "
                    << debug
                    << std::endl;
            }

            g_error_free(error);
            g_free(debug);

            break;
        }

        case GST_MESSAGE_EOS:
        {
            std::cout
                << "\nEOS received."
                << std::endl;

            g_main_loop_quit(loop);

            break;
        }

        default:
            break;
    }

    return G_SOURCE_CONTINUE;
}


static gboolean stop_pipeline(
    gpointer user_data)
{
    g_main_loop_quit(
        static_cast<GMainLoop *>(user_data));

    return G_SOURCE_REMOVE;
}


/*
 * ============================================================
 * Tee helper
 * ============================================================
 */

static bool link_tee_branch(
    GstElement *tee,
    GstElement *target,
    const char *branch_name)
{
    GstPad *tee_src =
        gst_element_request_pad_simple(
            tee,
            "src_%u");

    GstPad *sink_pad =
        gst_element_get_static_pad(
            target,
            "sink");

    if (!tee_src || !sink_pad)
    {
        g_printerr(
            "Failed to get pads for tee branch: %s\n",
            branch_name);

        if (tee_src)
            gst_object_unref(tee_src);

        if (sink_pad)
            gst_object_unref(sink_pad);

        return false;
    }

    bool ok =
        (gst_pad_link(
            tee_src,
            sink_pad)
         == GST_PAD_LINK_OK);

    if (!ok)
    {
        g_printerr(
            "Failed to link tee -> %s\n",
            branch_name);
    }

    gst_object_unref(tee_src);
    gst_object_unref(sink_pad);

    return ok;
}


/*
 * ============================================================
 * MAIN
 * ============================================================
 */

int main(int argc, char *argv[])
{
    /*
     * ========================================================
     * NEW:
     * Get RTSP URL from cameras.json
     *
     * No RTSP URL is accepted from command line anymore.
     * ========================================================
     */

    std::string rtsp_url_string =
        get_rtsp_url_from_json();

    if (rtsp_url_string.empty())
    {
        std::cerr
            << "ERROR: No valid RTSP stream available."
            << std::endl;

        return 1;
    }

    const gchar *rtsp_url =
        rtsp_url_string.c_str();


    gst_init(&argc, &argv);


    if (g_mkdir_with_parents(
            SNAPSHOT_DIR,
            0755) == 0)
    {
        g_print(
            "Snapshot directory ready: %s\n",
            SNAPSHOT_DIR);
    }
    else
    {
        g_printerr(
            "Warning: could not create snapshot directory: %s\n",
            SNAPSHOT_DIR);
    }


    if (g_mkdir_with_parents(
            EVENT_DIR,
            0755) == 0)
    {
        g_print(
            "Detection event directory ready: %s\n",
            EVENT_DIR);
    }
    else
    {
        g_printerr(
            "Warning: could not create event directory: %s\n",
            EVENT_DIR);
    }


    main_loop =
        g_main_loop_new(
            nullptr,
            FALSE);


    GstElement *pipeline =
        gst_pipeline_new(
            "yolo26-merged-pipeline");


    /* Core chain */

    GstElement *source =
        gst_element_factory_make(
            "nvurisrcbin",
            "camera-source");

    GstElement *streammux =
        gst_element_factory_make(
            "nvstreammux",
            "stream-muxer");

    GstElement *infer =
        gst_element_factory_make(
            "nvinfer",
            "yolo26-inference");

    GstElement *osd =
        gst_element_factory_make(
            "nvdsosd",
            "shared-osd");

    GstElement *tee =
        gst_element_factory_make(
            "tee",
            "infer-tee");


    /* Branch A: detection-only sink */

    GstElement *detect_queue =
        gst_element_factory_make(
            "queue",
            "detect-queue");

    GstElement *fake_sink =
        gst_element_factory_make(
            "fakesink",
            "detect-sink");


    /* Branch B: snapshot */

    GstElement *snapshot_queue =
        gst_element_factory_make(
            "queue",
            "snapshot-queue");

    GstElement *converter =
        gst_element_factory_make(
            "nvvideoconvert",
            "snapshot-converter");

    GstElement *snapshot_capsfilter =
        gst_element_factory_make(
            "capsfilter",
            "snapshot-caps");

    GstElement *jpegenc =
        gst_element_factory_make(
            "nvjpegenc",
            "snapshot-jpegenc");

    GstElement *appsink_element =
        gst_element_factory_make(
            "appsink",
            "snapshot-appsink");


    /* Branch C: live RTSP re-stream */

    GstElement *rtsp_queue =
        gst_element_factory_make(
            "queue",
            "rtsp-queue");

    GstElement *rtsp_sink =
        gst_element_factory_make(
            "nvrtspoutsinkbin",
            "rtsp-output");


    if (!pipeline ||
        !source ||
        !streammux ||
        !infer ||
        !osd ||
        !tee ||
        !detect_queue ||
        !fake_sink ||
        !snapshot_queue ||
        !converter ||
        !snapshot_capsfilter ||
        !jpegenc ||
        !appsink_element ||
        !rtsp_queue ||
        !rtsp_sink)
    {
        std::cerr
            << "ERROR: Failed to create one or more "
               "GStreamer/DeepStream elements."
            << std::endl;

        if (pipeline)
            gst_object_unref(pipeline);

        g_main_loop_unref(main_loop);

        return 1;
    }


    /* --- Properties --- */

    g_object_set(
        G_OBJECT(source),
        "uri",
        rtsp_url,
        "latency",
        200,
        nullptr);


    g_object_set(
        G_OBJECT(streammux),
        "batch-size",
        1,
        "width",
        1920,
        "height",
        1080,
        "live-source",
        TRUE,
        "batched-push-timeout",
        40000,
        nullptr);


    g_object_set(
        G_OBJECT(infer),
        "config-file-path",
        DEEPSTREAM_CONFIG,
        nullptr);


    g_object_set(
        G_OBJECT(osd),
        "process-mode",
        1,
        "display-bbox",
        TRUE,
        "display-text",
        TRUE,
        nullptr);


    g_object_set(
        G_OBJECT(appsink_element),
        "emit-signals",
        TRUE,
        "sync",
        FALSE,
        "max-buffers",
        1,
        "drop",
        TRUE,
        nullptr);


    GstCaps *snapshot_caps =
        gst_caps_from_string(
            "video/x-raw(memory:NVMM),"
            "format=NV12,"
            "width=1280,"
            "height=720");

    if (!snapshot_caps)
    {
        std::cerr
            << "ERROR: Failed to create snapshot caps."
            << std::endl;

        gst_object_unref(pipeline);
        g_main_loop_unref(main_loop);

        return 1;
    }


    g_object_set(
        G_OBJECT(snapshot_capsfilter),
        "caps",
        snapshot_caps,
        nullptr);

    gst_caps_unref(snapshot_caps);


    g_object_set(
        G_OBJECT(rtsp_sink),
        "idrinterval",
        30,
        "iframeinterval",
        30,
        nullptr);


    /* --- Add everything to pipeline --- */

    gst_bin_add_many(
        GST_BIN(pipeline),

        source,
        streammux,
        infer,
        osd,
        tee,

        detect_queue,
        fake_sink,

        snapshot_queue,
        converter,
        snapshot_capsfilter,
        jpegenc,
        appsink_element,

        rtsp_queue,
        rtsp_sink,

        nullptr);


    /* --- Static links --- */

    if (!gst_element_link(
            streammux,
            infer) ||

        !gst_element_link_many(
            infer,
            osd,
            tee,
            nullptr))
    {
        std::cerr
            << "ERROR: Failed to link "
               "streammux -> nvinfer -> "
               "nvdsosd -> tee."
            << std::endl;

        gst_object_unref(pipeline);
        g_main_loop_unref(main_loop);

        return 1;
    }


    if (!gst_element_link_many(
            detect_queue,
            fake_sink,
            nullptr))
    {
        std::cerr
            << "ERROR: Failed to link "
               "detection branch."
            << std::endl;

        gst_object_unref(pipeline);
        g_main_loop_unref(main_loop);

        return 1;
    }


    if (!gst_element_link_many(
            snapshot_queue,
            converter,
            snapshot_capsfilter,
            jpegenc,
            appsink_element,
            nullptr))
    {
        std::cerr
            << "ERROR: Failed to link "
               "snapshot branch."
            << std::endl;

        gst_object_unref(pipeline);
        g_main_loop_unref(main_loop);

        return 1;
    }


    /* rtsp_queue -> nvrtspoutsinkbin */

    {
        GstPad *rtsp_sink_pad =
            gst_element_request_pad_simple(
                rtsp_sink,
                "vsink");

        GstPad *queue_src_pad =
            gst_element_get_static_pad(
                rtsp_queue,
                "src");

        if (!rtsp_sink_pad ||
            !queue_src_pad ||
            gst_pad_link(
                queue_src_pad,
                rtsp_sink_pad)
                != GST_PAD_LINK_OK)
        {
            std::cerr
                << "ERROR: Failed to link "
                   "rtsp-queue -> "
                   "nvrtspoutsinkbin."
                << std::endl;

            if (rtsp_sink_pad)
                gst_object_unref(
                    rtsp_sink_pad);

            if (queue_src_pad)
                gst_object_unref(
                    queue_src_pad);

            gst_object_unref(pipeline);
            g_main_loop_unref(main_loop);

            return 1;
        }

        gst_object_unref(rtsp_sink_pad);
        gst_object_unref(queue_src_pad);
    }


    /* --- Tee: request 3 src pads --- */

    if (!link_tee_branch(
            tee,
            detect_queue,
            "detection queue") ||

        !link_tee_branch(
            tee,
            snapshot_queue,
            "snapshot queue") ||

        !link_tee_branch(
            tee,
            rtsp_queue,
            "rtsp queue"))
    {
        gst_object_unref(pipeline);
        g_main_loop_unref(main_loop);

        return 1;
    }


    /* --- Dynamic source pad --- */

    g_signal_connect(
        source,
        "pad-added",
        G_CALLBACK(on_pad_added),
        streammux);


    /* --- Detection probe --- */

    GstPad *osd_src =
        gst_element_get_static_pad(
            osd,
            "src");

    if (!osd_src)
    {
        std::cerr
            << "ERROR: Failed to get "
               "shared nvdsosd src pad."
            << std::endl;

        gst_object_unref(pipeline);
        g_main_loop_unref(main_loop);

        return 1;
    }


    gst_pad_add_probe(
        osd_src,
        GST_PAD_PROBE_TYPE_BUFFER,
        infer_src_probe,
        nullptr,
        nullptr);

    gst_object_unref(osd_src);


    /* --- appsink callback --- */

    g_signal_connect(
        GST_APP_SINK(appsink_element),
        "new-sample",
        G_CALLBACK(on_new_sample),
        nullptr);


    /* --- Bus --- */

    GstBus *bus =
        gst_element_get_bus(pipeline);

    gst_bus_add_watch(
        bus,
        bus_callback,
        main_loop);

    gst_object_unref(bus);


    /* --- Ctrl+C --- */

    g_unix_signal_add(
        SIGINT,
        stop_pipeline,
        main_loop);


    /* --- Print configuration --- */

    g_print(
        "\n========================================\n");

    g_print(
        " YOLO26 MERGED PIPELINE\n");

    g_print(
        " CAMERA JSON: %s\n",
        CAMERA_JSON_FILE);

    g_print(
        " RTSP INPUT:  %s\n",
        rtsp_url);

    g_print(
        " RTSP OUTPUT: %s\n",
        RTSP_OUTPUT_URL_HINT);

    g_print(
        " DeepStream config: %s\n",
        DEEPSTREAM_CONFIG);

    g_print(
        " Snapshot directory: %s\n",
        SNAPSHOT_DIR);

    g_print(
        " Event directory: %s\n",
        EVENT_DIR);

    g_print(
        " Snapshot cooldown: %d seconds\n",
        SNAPSHOT_COOLDOWN_SECONDS);

    g_print(
        " Branches: detection-only | "
        "snapshot+event | live RTSP out\n");

    g_print(
        "========================================\n\n");


    /* --- Start pipeline --- */

    GstStateChangeReturn state =
        gst_element_set_state(
            pipeline,
            GST_STATE_PLAYING);

    if (state == GST_STATE_CHANGE_FAILURE)
    {
        std::cerr
            << "ERROR: Failed to start pipeline."
            << std::endl;

        gst_element_set_state(
            pipeline,
            GST_STATE_NULL);

        gst_object_unref(pipeline);
        g_main_loop_unref(main_loop);

        return 1;
    }


    g_main_loop_run(main_loop);


    /* --- Cleanup --- */

    gst_element_set_state(
        pipeline,
        GST_STATE_NULL);

    gst_object_unref(pipeline);

    g_main_loop_unref(main_loop);

    std::cout
        << "Pipeline stopped."
        << std::endl;

    return 0;
}
