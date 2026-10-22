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
 * SNAPSHOTS:
 *   Red-light snapshots retain the existing line-crossing behavior.
 *   Wrong-route snapshots are triggered independently by dynamic motion
 *   analysis and do not require a signal state or crosswalk line.
 *   The JPG is a CLEAN CROP of just that vehicle (no boxes, labels or
 *   red line), cut from the exact frame that confirmed the crossing,
 *   at native pixels, before nvdsosd draws anything.
 *   A crossing needs LINE_HYSTERESIS_PX clearance and
 *   CROSS_CONFIRM_FRAMES consecutive frames. Only the pixel crop runs
 *   on the video thread; JPEG completion, verification and the JSON
 *   event run on a worker thread. A failed save is retried (with
 *   back-off) for as long as the vehicle is in view, and never
 *   produces a JSON event.
 *
 * Pipeline:
 *
 *   nvurisrcbin -> streammux -> nvinfer -> nvtracker -> nvdsosd -> tee
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
#include <cmath>
#include <deque>
#include <algorithm>
#include <ctime>
#include <chrono>
#include <csignal>
#include <string>
#include <map>
#include <mutex>
#include <vector>
#include <set>
#include <unordered_map>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <nlohmann/json.hpp>

#include "gstnvdsmeta.h"
#include "nvdsmeta.h"
#include "nvds_version.h"
#include "nvbufsurface.h"
#include "nvds_obj_encode.h"      /* DeepStream crop + JPEG encode (link: see build note) */

/* nvds_obj_enc_create_context() takes a GPU id on DeepStream 7+. */
#if defined(NVDS_VERSION_MAJOR) && (NVDS_VERSION_MAJOR >= 7)
#define OBJ_ENC_CREATE_CONTEXT()  nvds_obj_enc_create_context(0)
#else
#define OBJ_ENC_CREATE_CONTEXT()  nvds_obj_enc_create_context()
#endif

using json = nlohmann::json;


/*
 * ============================================================
 * Configuration
 * ============================================================
 */

#define SNAPSHOT_DIR              "/root/video_recorder/snapshots"
#define EVENT_DIR                 "/root/video_recorder/events/pending"
#define DEEPSTREAM_CONFIG         "/opt/radian/ai/config/config_infer_primary_yolo26.txt"
#define VEHICLE_SNAPSHOT_DIR      SNAPSHOT_DIR "/vehicles"

/* Must match nvstreammux width/height in main() */
#define MUX_WIDTH                 1920
#define MUX_HEIGHT                1080

/* nvtracker (needed to give each vehicle a stable ID) */
#define TRACKER_LIB   "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
#define TRACKER_CONFIG "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"
#define TRACKER_WIDTH  640   /* must be multiple of 32 */
#define TRACKER_HEIGHT 384   /* must be multiple of 32 */

/* Vehicle crop settings */
#define CROP_PADDING_PERCENT      20     /* margin around the vehicle box; 0 = exact box */
#define SNAPSHOT_JPEG_QUALITY     100    /* 0..100: only limits EXTRA compression; real detail is set by camera, resolution and vehicle size */
#define SNAPSHOT_MIN_CROP_PX      32     /* smaller crops are rejected and retried later */
#define SNAPSHOT_SIZE_TOLERANCE   8      /* allowed |saved - requested| size in px */

/* Line-crossing logic */
#define LINE_HYSTERESIS_PX        15     /* vehicle must be this far past the line */
#define CROSS_CONFIRM_FRAMES      3      /* consecutive frames on the new side */
#define SNAPSHOT_RETRY_GAP_FRAMES     5  /* first retry gap; doubles after each failure */
#define SNAPSHOT_RETRY_MAX_GAP_FRAMES 60 /* back-off ceiling (retries never stop while visible) */
#define ID_REUSE_GAP_FRAMES       450    /* same tracker id after this gap = a new vehicle */

/* Dynamic wrong-route detection: thresholds are temporal/motion confidence,
 * not fixed screen directions or lane coordinates. */
#define WRONG_ROUTE_HISTORY_POINTS       12
#define WRONG_ROUTE_MIN_HISTORY_FRAMES   8
#define WRONG_ROUTE_MIN_DISPLACEMENT_PX  20.0f
#define WRONG_ROUTE_CONFIRM_FRAMES       8
#define WRONG_ROUTE_MIN_FLOW_TRACKS      3
#define WRONG_ROUTE_MIN_FLOW_STRENGTH    0.25f
#define WRONG_ROUTE_OPPOSITE_DOT         -0.70f
#define WRONG_ROUTE_MAX_MOTION_AGE       12
#define RTSP_OUTPUT_URL_HINT      "rtsp://192.168.1.155:8554/ai"

/* Camera discovery output */
#define CAMERA_JSON_FILE          "/var/lib/camera-discovery/cameras.json"
#define CROSSWALK_JSON_FILE "/var/lib/radian/crosswalk/crosswalk.json"


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

/*
 * Line-crossing state. Only touched from the single streaming thread
 * that runs infer_src_probe (the save worker reports back through
 * save_results, which that probe drains), so no extra locking is
 * needed here.
 */
enum class CaptureState
{
    Idle,      /* nothing requested                                   */
    Pending,   /* a save is running (reserved: no duplicate job)      */
    Failed,    /* last save failed -> eligible for retry              */
    Saved      /* jpg + event written (key also kept in captured_keys)*/
};

struct MotionPoint
{
    float x = 0.0f;
    float y = 0.0f;
    int frame = 0;
};

struct TrackState
{
    float last_y = 0.0f;       /* bottom edge of the bbox in previous frame */
    int last_frame = 0;

    int side = 0;              /* confirmed side: -1 above, +1 below, 0 unknown */
    int cand_side = 0;         /* side being confirmed                          */
    int cand_count = 0;        /* consecutive frames seen on cand_side          */

    /* Per-track motion history used by dynamic wrong-route estimation. */
    std::deque<MotionPoint> motion_history;
    float motion_dx = 0.0f;
    float motion_dy = 0.0f;
    bool motion_valid = false;
    int motion_last_frame = -1;
    int opposite_count = 0;
    bool wrong_route_confirmed = false;
    bool wrong_route_waiting = false;
    bool wrong_route_lost_logged = false;
    CaptureState wrong_route_cap = CaptureState::Idle;
    int wrong_route_attempts = 0;
    int wrong_route_next_retry_frame = 0;

    bool crossed_waiting = false;   /* crossing confirmed, jpg not saved yet */
    bool lost_logged = false;       /* "left view before saving" already logged */
    CaptureState cap = CaptureState::Idle;
    int attempts = 0;
    int next_retry_frame = 0;
};

/*
 * Capture key = (era, tracker id), era = (tracker epoch << 20) | id
 * generation. The epoch increases on a detected stream/tracker restart;
 * the generation increases when the same tracker id comes back after a
 * long gap (id reuse). Keys are NEVER erased.
 */
typedef std::pair<guint64, guint64> CaptureKey;

static std::unordered_map<guint64, TrackState> track_states;
static std::unordered_map<guint64, guint32> id_generation;
static std::set<CaptureKey> captured_keys;  /* red-light/line-crossing captures */
static std::set<CaptureKey> wrong_route_captured_keys;
static guint64 tracker_epoch = 0;
static int last_seen_frame_num = -1;
static guint64 last_seen_pts = 0;

/* bumped from on_pad_added every time the camera source is (re)linked */
static gint source_link_count = 0;
static gint source_link_seen = 0;

/* DeepStream object crop/encode context (created in main) */
static NvDsObjEncCtxHandle obj_enc_ctx = nullptr;

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

struct CrosswalkLineConfig
{
    bool detected = false;
    int line_y = -1;
    int frame_width = 0;
    int frame_height = 0;
};


static CrosswalkLineConfig crosswalk_config;

/* Lane 1 signal state, refreshed from disk at most once per second. */
#define TRAFFIC_SIGNAL_JSON_FILE "/opt/telemetry-app/traffic_signal_state.json"
static std::string lane_1_signal = "UNKNOWN";
static std::chrono::steady_clock::time_point last_signal_state_read =
    std::chrono::steady_clock::time_point::min();

static void refresh_lane_1_signal_state()
{
    const auto now = std::chrono::steady_clock::now();
    if (last_signal_state_read != std::chrono::steady_clock::time_point::min() &&
        now - last_signal_state_read < std::chrono::seconds(1))
        return;

    last_signal_state_read = now;
    std::string new_state = "UNKNOWN";

    try
    {
        std::ifstream file(TRAFFIC_SIGNAL_JSON_FILE);
        if (!file.is_open())
        {
            g_printerr("[SIGNAL] Cannot open %s; snapshots disabled until RED is confirmed\n",
                       TRAFFIC_SIGNAL_JSON_FILE);
            lane_1_signal = "UNKNOWN";
            return;
        }

        json data;
        file >> data;
        if (data.is_object() && data.contains("lane_1") &&
            data["lane_1"].is_string())
        {
            std::string value = data["lane_1"].get<std::string>();
            for (char &c : value)
                c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));

            if (value == "RED" || value == "GREEN")
                new_state = value;
        }

        if (new_state == "UNKNOWN")
            g_printerr("[SIGNAL] lane_1 missing or invalid in %s; snapshots disabled\n",
                       TRAFFIC_SIGNAL_JSON_FILE);

        if (new_state != lane_1_signal)
            g_print("[SIGNAL] Lane 1 state: %s\n", new_state.c_str());

        lane_1_signal = new_state;
    }
    catch (const std::exception &e)
    {
        lane_1_signal = "UNKNOWN";
        g_printerr("[SIGNAL] Failed to parse %s: %s; snapshots disabled\n",
                   TRAFFIC_SIGNAL_JSON_FILE, e.what());
    }
}

static bool load_crosswalk_line_config()
{
    std::ifstream file(CROSSWALK_JSON_FILE);
    if (!file.is_open())
    {
        std::cerr << "[CROSSWALK] Cannot open " << CROSSWALK_JSON_FILE << std::endl;
        return false;
    }

    try
    {
        nlohmann::json j;
        file >> j;

        if (!j.contains("crosswalk_detected") ||
            !j.contains("line_y") ||
            !j.contains("frame_width") ||
            !j.contains("frame_height"))
        {
            std::cerr << "[CROSSWALK] Invalid JSON: required fields missing" << std::endl;
            return false;
        }

        crosswalk_config.detected = j["crosswalk_detected"].get<bool>();
        crosswalk_config.line_y = j["line_y"].get<int>();
        crosswalk_config.frame_width = j["frame_width"].get<int>();
        crosswalk_config.frame_height = j["frame_height"].get<int>();

        std::cout << "[CROSSWALK] JSON loaded" << std::endl;
        std::cout << "[CROSSWALK] detected = " << crosswalk_config.detected << std::endl;
        std::cout << "[CROSSWALK] line_y = " << crosswalk_config.line_y << std::endl;
        std::cout << "[CROSSWALK] frame = "
                  << crosswalk_config.frame_width << "x"
                  << crosswalk_config.frame_height << std::endl;

        if (!crosswalk_config.detected ||
            crosswalk_config.line_y < 0 ||
            crosswalk_config.frame_width <= 0 ||
            crosswalk_config.frame_height <= 0)
        {
            std::cerr << "[CROSSWALK] No valid crosswalk line configuration" << std::endl;
            return false;
        }

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[CROSSWALK] JSON parse error: " << e.what() << std::endl;
        return false;
    }
}


/*
 * Y position of the red line in the pipeline (streammux) coordinate
 * space. Object boxes are in this same space, so drawing and crossing
 * detection always agree.
 */
static int compute_line_y(const NvDsFrameMeta *frame_meta)
{
    int h = (frame_meta->pipeline_height > 0)
        ? static_cast<int>(frame_meta->pipeline_height)
        : MUX_HEIGHT;

    int y = crosswalk_config.line_y;

    if (crosswalk_config.frame_height > 0 &&
        crosswalk_config.frame_height != h)
    {
        y = static_cast<int>(
            static_cast<double>(crosswalk_config.line_y) *
            static_cast<double>(h) /
            static_cast<double>(crosswalk_config.frame_height));
    }

    if (y < 0)
        y = 0;

    if (y >= h)
        y = h - 1;

    return y;
}


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


/*
 * First free path "<dir>/<stem><ext>", then "<stem>_1<ext>", ...
 * so a file is never overwritten.
 */
static std::string unique_path(
    const std::string &dir,
    const std::string &stem,
    const char *ext)
{
    std::string path = dir + "/" + stem + ext;

    for (int i = 1;
         i < 1000 && g_file_test(path.c_str(), G_FILE_TEST_EXISTS);
         ++i)
    {
        path = dir + "/" + stem + "_" + std::to_string(i) + ext;
    }

    return path;
}

/*
 * Signal-jumping snapshot names:
 *   NUMBERPLATE-SIGNAL-JUMPING.jpg
 *   NUMBERPLATE2-SIGNAL-JUMPING.jpg
 *   NUMBERPLATE3-SIGNAL-JUMPING.jpg
 * Keep numbering unique across service restarts by checking existing files.
 */
static std::string next_signal_jump_snapshot_path(const std::string &dir)
{
    const std::string first = dir + "/NUMBERPLATE-SIGNAL-JUMPING.jpg";
    if (!g_file_test(first.c_str(), G_FILE_TEST_EXISTS))
        return first;

    for (unsigned int number = 2; number < 1000000; ++number)
    {
        const std::string path = dir + "/NUMBERPLATE" +
            std::to_string(number) + "-SIGNAL-JUMPING.jpg";
        if (!g_file_test(path.c_str(), G_FILE_TEST_EXISTS))
            return path;
    }

    return std::string();
}

static std::string next_wrong_route_snapshot_path(const std::string &dir)
{
    const std::string first = dir + "/WRONG-ROUTE.jpg";
    if (!g_file_test(first.c_str(), G_FILE_TEST_EXISTS))
        return first;

    for (unsigned int number = 2; number < 1000000; ++number)
    {
        const std::string path = dir + "/WRONG-ROUTE-" +
            std::to_string(number) + ".jpg";
        if (!g_file_test(path.c_str(), G_FILE_TEST_EXISTS))
            return path;
    }
    return std::string();
}


/* "id12", "id12_e1" or "id12_e1_g2" (epoch / generation only after a reset or id reuse) */
static std::string track_tag(guint64 track_id, guint64 epoch, guint32 gen)
{
    std::ostringstream oss;

    oss << "id" << track_id;

    if (epoch > 0)
        oss << "_e" << epoch;

    if (gen > 0)
        oss << "_g" << gen;

    return oss.str();
}


/*
 * Read width/height from a JPEG's SOF marker, so we verify what was
 * really saved instead of trusting what we asked for.
 */
static bool read_jpeg_size(
    const std::string &path,
    int &width,
    int &height)
{
    std::ifstream f(path, std::ios::binary);

    if (!f)
        return false;

    if (f.get() != 0xFF || f.get() != 0xD8)
        return false;

    while (f)
    {
        int c = f.get();

        while (c != EOF && c != 0xFF)
            c = f.get();

        if (c == EOF)
            return false;

        int m = f.get();

        while (m == 0xFF)
            m = f.get();

        if (m == EOF)
            return false;

        /* markers without a length field */
        if (m == 0x01 || (m >= 0xD0 && m <= 0xD8))
            continue;

        /* end of image / start of scan before any frame header */
        if (m == 0xD9 || m == 0xDA)
            return false;

        int hi = f.get();
        int lo = f.get();

        if (lo == EOF)
            return false;

        const int len = (hi << 8) | lo;

        const bool is_sof =
            (m >= 0xC0 && m <= 0xCF) &&
            m != 0xC4 && m != 0xC8 && m != 0xCC;

        if (is_sof)
        {
            f.get();                       /* sample precision */

            int h1 = f.get();
            int h2 = f.get();
            int w1 = f.get();
            int w2 = f.get();

            if (w2 == EOF)
                return false;

            height = (h1 << 8) | h2;
            width  = (w1 << 8) | w2;

            return width > 0 && height > 0;
        }

        f.seekg(len - 2, std::ios::cur);
    }

    return false;
}


/*
 * Check a saved jpg: complete (starts with SOI, ends with EOI), has a
 * readable frame header, and report its real size.
 */
static bool inspect_jpeg(
    const std::string &path,
    int &width,
    int &height,
    long &bytes)
{
    {
        std::ifstream f(path, std::ios::binary | std::ios::ate);

        if (!f)
            return false;

        const long size = static_cast<long>(f.tellg());

        if (size < 4)
            return false;

        bytes = size;

        const long tail = (size < 64) ? size : 64;
        std::vector<unsigned char> t(static_cast<size_t>(tail));

        f.seekg(size - tail);
        f.read(reinterpret_cast<char *>(t.data()), tail);

        if (!f)
            return false;

        bool has_eoi = false;

        for (long i = 0; i + 1 < tail; ++i)
        {
            if (t[i] == 0xFF && t[i + 1] == 0xD9)
                has_eoi = true;
        }

        if (!has_eoi)
            return false;       /* truncated / still being written */
    }

    return read_jpeg_size(path, width, height);
}


static bool save_detection_event(
    const std::string &object_name,
    float confidence,
    std::time_t detection_time,
    const std::string &timestamp,
    guint64 track_id,
    guint64 epoch,
    guint32 gen,
    const std::string &snapshot_path,
    const std::string &violation_type)
{
    const std::string filename = unique_path(
        EVENT_DIR,
        safe_filename_component(violation_type) + "_" +
            safe_filename_component(object_name) + "_" +
            track_tag(track_id, epoch, gen) + "_" + timestamp,
        ".json");

    std::ofstream file(filename);

    if (!file)
    {
        g_printerr(
            "Failed to create detection event: %s\n",
            filename.c_str());

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

         << "  \"violation_type\": \""
         << json_escape(violation_type)
         << "\",\n"

         << "  \"snapshot\": \""
         << json_escape(snapshot_path)
         << "\"\n"

         << "}\n";

    file.close();

    if (!file)
    {
        g_printerr(
            "Failed while writing detection event: %s\n",
            filename.c_str());

        std::remove(filename.c_str());
        return false;
    }

    g_print(
        "EVENT SAVED: %s\n",
        filename.c_str());

    return true;
}


/*
 * ============================================================
 * Vehicle snapshot: crop on the video thread, finish on a worker
 * ============================================================
 *
 * Split in two on purpose:
 *
 *   start_vehicle_crop()  (video thread, before nvdsosd)
 *       Cuts the vehicle's pixels out of the CURRENT frame and hands
 *       them to the DeepStream encoder. This must happen here, because
 *       nvdsosd draws on the same buffer right afterwards.
 *
 *   save_worker_thread()  (worker thread)
 *       Waits for the encoder to finish writing, verifies the jpg and
 *       writes the JSON event, then reports back. Slow disk or slow
 *       JPEG encoding therefore never blocks the video pipeline.
 *
 * Only ONE job runs at a time, so the encoder context is never used
 * from two threads at once.
 */

struct SaveJob
{
    std::string object_name;
    float confidence = 0.0f;
    guint64 track_id = 0;
    guint64 epoch = 0;
    guint32 gen = 0;
    CaptureKey key;
    std::string violation_type = "red_light_violation";

    std::time_t now = 0;
    std::string timestamp;
    std::string jpg_path;

    int cw = 0, ch = 0;        /* requested crop size */
};

struct SaveResult
{
    CaptureKey key;
    guint64 track_id = 0;
    std::string violation_type = "red_light_violation";
    bool success = false;
    bool encoder_related = false;   /* encoder / jpg problem, not "crop too small" */
};

static std::mutex save_mutex;
static std::vector<SaveResult> save_results;
static bool save_in_flight = false;
static bool encoder_validated_logged = false;   /* worker thread only */


static bool save_job_in_flight()
{
    std::lock_guard<std::mutex> lock(save_mutex);
    return save_in_flight;
}


static void wait_for_save_jobs()
{
    for (int i = 0; i < 100 && save_job_in_flight(); ++i)
        g_usleep(50 * 1000);
}


static gpointer save_worker_thread(gpointer data)
{
    SaveJob *job = static_cast<SaveJob *>(data);

    bool ok = false;
    bool encoder_related = false;

    do
    {
        /* wait until the encoder has written the file */
        nvds_obj_enc_finish(obj_enc_ctx);

        int jw = 0, jh = 0;
        long bytes = 0;

        if (!inspect_jpeg(job->jpg_path, jw, jh, bytes))
        {
            g_printerr(
                "Snapshot failed id=%llu: saved file missing, truncated "
                "or not a valid jpeg: %s\n",
                static_cast<unsigned long long>(job->track_id),
                job->jpg_path.c_str());

            encoder_related = true;
            break;
        }

        if (std::abs(jw - job->cw) > SNAPSHOT_SIZE_TOLERANCE ||
            std::abs(jh - job->ch) > SNAPSHOT_SIZE_TOLERANCE)
        {
            g_printerr(
                "Snapshot failed id=%llu: saved %dx%d but requested "
                "%dx%d (not a vehicle-only crop)\n",
                static_cast<unsigned long long>(job->track_id),
                jw, jh, job->cw, job->ch);

            encoder_related = true;
            break;
        }

        /* a real vehicle photo is never this small: warn, do not reject */
        if (bytes < static_cast<long>(jw) * jh / 20)
        {
            g_printerr(
                "WARNING id=%llu: jpg is only %ld bytes for %dx%d "
                "(possibly blank crop) - please look at it: %s\n",
                static_cast<unsigned long long>(job->track_id),
                bytes, jw, jh, job->jpg_path.c_str());
        }

        g_print(
            "SNAPSHOT SAVED: %s (%dx%d, %ld bytes)\n",
            job->jpg_path.c_str(), jw, jh, bytes);

        if (!save_detection_event(
                job->object_name,
                job->confidence,
                job->now,
                job->timestamp,
                job->track_id,
                job->epoch,
                job->gen,
                job->jpg_path,
                job->violation_type))
        {
            break;
        }

        if (!encoder_validated_logged)
        {
            encoder_validated_logged = true;

            g_print(
                "ENCODER VALIDATED: first snapshot is a complete %dx%d jpeg. "
                "Open it once to confirm the crop looks right: %s\n",
                jw, jh, job->jpg_path.c_str());
        }

        ok = true;
    }
    while (false);

    if (!ok)
        std::remove(job->jpg_path.c_str());   /* no orphan / partial image */

    {
        std::lock_guard<std::mutex> lock(save_mutex);

        SaveResult r;
        r.key = job->key;
        r.track_id = job->track_id;
        r.violation_type = job->violation_type;
        r.success = ok;
        r.encoder_related = encoder_related;

        save_results.push_back(r);
        save_in_flight = false;
    }

    delete job;
    return nullptr;
}


/*
 * Crop the vehicle out of the frame that is flowing through the probe
 * RIGHT NOW (clean, before nvdsosd) and start the encode.
 *
 * Returns true if a job was started (its result arrives later through
 * save_results). Returns false on an immediate failure, and sets
 * encoder_related when the cause is the encoder rather than the crop.
 * There is no full-frame fallback.
 */
static bool start_vehicle_crop(
    GstBuffer *buf,
    NvDsFrameMeta *frame_meta,
    NvDsObjectMeta *obj_meta,
    const std::string &object_name,
    float confidence,
    guint64 track_id,
    guint64 epoch,
    guint32 gen,
    const CaptureKey &key,
    const std::string &violation_type,
    bool &encoder_related)
{
    encoder_related = false;

    if (!obj_enc_ctx)
    {
        g_printerr("Snapshot skipped: object encoder not available\n");
        return false;
    }

    GstMapInfo map;

    if (!gst_buffer_map(buf, &map, GST_MAP_READ))
    {
        g_printerr("Snapshot failed: cannot map frame buffer\n");
        return false;
    }

    NvBufSurface *surface = reinterpret_cast<NvBufSurface *>(map.data);

    const NvOSD_RectParams original = obj_meta->rect_params;

    bool started = false;
    SaveJob *job = nullptr;
    std::string jpg_path;

    do
    {
        /* ---- real size of the frame we are cropping from ---- */
        if (!surface ||
            frame_meta->batch_id >= surface->numFilled)
        {
            g_printerr(
                "Snapshot failed: bad batch id %u\n",
                frame_meta->batch_id);
            break;
        }

        const float surf_w =
            static_cast<float>(surface->surfaceList[frame_meta->batch_id].width);
        const float surf_h =
            static_cast<float>(surface->surfaceList[frame_meta->batch_id].height);

        const float pipe_w = (frame_meta->pipeline_width > 0)
            ? static_cast<float>(frame_meta->pipeline_width) : surf_w;
        const float pipe_h = (frame_meta->pipeline_height > 0)
            ? static_cast<float>(frame_meta->pipeline_height) : surf_h;

        if (surf_w <= 0 || surf_h <= 0)
        {
            g_printerr("Snapshot failed: invalid surface size\n");
            break;
        }

        /* box (pipeline coordinates) -> surface pixels, clamped to the frame */
        const float sx = surf_w / pipe_w;
        const float sy = surf_h / pipe_h;

        float bx1 = original.left * sx;
        float by1 = original.top * sy;
        float bx2 = (original.left + original.width) * sx;
        float by2 = (original.top + original.height) * sy;

        if (bx1 < 0) bx1 = 0;
        if (by1 < 0) by1 = 0;
        if (bx2 > surf_w) bx2 = surf_w;
        if (by2 > surf_h) by2 = surf_h;

        const float bw = bx2 - bx1;
        const float bh = by2 - by1;

        /* 20% padding, clamped to the FRAME only (never to the line) */
        const float pad_x = bw * CROP_PADDING_PERCENT / 100.0f;
        const float pad_y = bh * CROP_PADDING_PERCENT / 100.0f;

        float x1 = bx1 - pad_x;
        float y1 = by1 - pad_y;
        float x2 = bx2 + pad_x;
        float y2 = by2 + pad_y;

        if (x1 < 0) x1 = 0;
        if (y1 < 0) y1 = 0;
        if (x2 > surf_w) x2 = surf_w;
        if (y2 > surf_h) y2 = surf_h;

        const int cx = static_cast<int>(x1) & ~1;
        const int cy = static_cast<int>(y1) & ~1;
        const int cw = static_cast<int>(x2 - cx) & ~1;
        const int ch = static_cast<int>(y2 - cy) & ~1;

        if (cw < SNAPSHOT_MIN_CROP_PX || ch < SNAPSHOT_MIN_CROP_PX)
        {
            g_printerr(
                "Snapshot rejected id=%llu: crop %dx%d too small "
                "(surface %.0fx%.0f, box %.0f,%.0f %.0fx%.0f)\n",
                static_cast<unsigned long long>(track_id),
                cw, ch, surf_w, surf_h, bx1, by1, bw, bh);
            break;
        }

        /* the crop must contain the whole vehicle box */
        if (cx > bx1 + 2 || cy > by1 + 2 ||
            cx + cw < bx2 - 2 || cy + ch < by2 - 2)
        {
            g_printerr(
                "Snapshot rejected id=%llu: crop %d,%d %dx%d does not "
                "contain vehicle box %.0f,%.0f-%.0f,%.0f\n",
                static_cast<unsigned long long>(track_id),
                cx, cy, cw, ch, bx1, by1, bx2, by2);
            break;
        }

        /* ---- file names ---- */
        if (g_mkdir_with_parents(VEHICLE_SNAPSHOT_DIR, 0755) != 0)
        {
            g_printerr(
                "Failed to create vehicle snapshot directory: %s\n",
                VEHICLE_SNAPSHOT_DIR);
            break;
        }

        const std::time_t now = std::time(nullptr);
        const std::string timestamp = get_timestamp();

        jpg_path = (violation_type == "wrong_route")
            ? next_wrong_route_snapshot_path(VEHICLE_SNAPSHOT_DIR)
            : next_signal_jump_snapshot_path(VEHICLE_SNAPSHOT_DIR);
        if (jpg_path.empty())
        {
            g_printerr("Failed to allocate a %s snapshot filename\n",
                       violation_type.c_str());
            break;
        }

        /* ---- crop at native pixels and queue the encode ---- */
        NvOSD_RectParams &r = obj_meta->rect_params;

        r.left   = static_cast<float>(cx);
        r.top    = static_cast<float>(cy);
        r.width  = static_cast<float>(cw);
        r.height = static_cast<float>(ch);

        NvDsObjEncUsrArgs args;
        memset(&args, 0, sizeof(args));

        args.saveImg = TRUE;
        args.attachUsrMeta = FALSE;
        args.scaleImg = FALSE;                 /* never resize */
        args.quality = SNAPSHOT_JPEG_QUALITY;
        args.objNum = 1;

        g_strlcpy(args.fileNameImg, jpg_path.c_str(), sizeof(args.fileNameImg));

        const gboolean enc_ok =
            nvds_obj_enc_process(
                obj_enc_ctx, &args, surface, obj_meta, frame_meta);

        obj_meta->rect_params = original;       /* OSD still draws the real box */

        if (!enc_ok)
        {
            g_printerr(
                "Snapshot failed id=%llu: nvds_obj_enc_process failed "
                "(crop %d,%d %dx%d of %.0fx%.0f)\n",
                static_cast<unsigned long long>(track_id),
                cx, cy, cw, ch, surf_w, surf_h);

            encoder_related = true;
            break;
        }

        job = new SaveJob();
        job->object_name = object_name;
        job->confidence = confidence;
        job->track_id = track_id;
        job->epoch = epoch;
        job->gen = gen;
        job->key = key;
        job->violation_type = violation_type;
        job->now = now;
        job->timestamp = timestamp;
        job->jpg_path = jpg_path;
        job->cw = cw;
        job->ch = ch;

        started = true;
    }
    while (false);

    obj_meta->rect_params = original;
    gst_buffer_unmap(buf, &map);

    if (!started)
        return false;

    {
        std::lock_guard<std::mutex> lock(save_mutex);
        save_in_flight = true;
    }

    GThread *t = g_thread_new("vehicle-save", save_worker_thread, job);
    g_thread_unref(t);   /* detach: the worker frees the job */

    return true;
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

    static bool caps_logged = false;

    if (!caps_logged)
    {
        caps_logged = true;

        GstCaps *c = gst_sample_get_caps(sample);
        gchar *c_str = c ? gst_caps_to_string(c) : nullptr;

        g_print(
            "Snapshot branch encoder output caps: %s\n",
            c_str ? c_str : "unknown");

        g_free(c_str);
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
 * Detection probe — red-light line-crossing + dynamic wrong-route detection
 * ============================================================
 *
 * Attached to the nvdsosd SINK pad, so the frame is still clean.
 *
 * Per tracked vehicle we follow the bottom edge of its box:
 *   - zone  -1 above / +1 below the line, 0 inside the +-15 px band
 *   - a crossing is confirmed only after CROSS_CONFIRM_FRAMES
 *     consecutive frames beyond the band on the opposite side
 *     (both directions are supported)
 *   - on confirmation the CURRENT frame is cropped and the save is
 *     handed to a worker; if it fails it is retried on later frames
 *     (with back-off, no attempt limit) while the vehicle stays in
 *     view, and the vehicle is marked captured only after the jpg and
 *     event were written.
 */

static bool is_vehicle_class(const std::string &name)
{
    static const std::set<std::string> vehicles = {
        "car", "motorcycle", "bus", "truck"
    };

    return vehicles.count(name) > 0;
}


static int consecutive_encoder_failures = 0;



/* Estimate the dominant live traffic direction from recent tracked motion.
 * Direction is learned from the scene; no fixed left/right/up/down direction
 * or lane coordinates are configured. A weak/ambiguous flow returns false. */
static bool estimate_dominant_flow(int frame_num, float &flow_x, float &flow_y,
                                   int &supporting_tracks, float &strength)
{
    double sum_x = 0.0;
    double sum_y = 0.0;
    supporting_tracks = 0;

    for (const auto &entry : track_states)
    {
        const TrackState &ts = entry.second;
        if (!ts.motion_valid ||
            frame_num - ts.last_frame > WRONG_ROUTE_MAX_MOTION_AGE ||
            frame_num - ts.motion_last_frame > WRONG_ROUTE_MAX_MOTION_AGE)
            continue;

        const double magnitude = std::hypot(ts.motion_dx, ts.motion_dy);
        if (magnitude < 1.0)
            continue;

        sum_x += ts.motion_dx / magnitude;
        sum_y += ts.motion_dy / magnitude;
        ++supporting_tracks;
    }

    if (supporting_tracks < WRONG_ROUTE_MIN_FLOW_TRACKS)
        return false;

    const double resultant = std::hypot(sum_x, sum_y);
    strength = static_cast<float>(resultant / supporting_tracks);
    if (strength < WRONG_ROUTE_MIN_FLOW_STRENGTH || resultant < 1e-6)
        return false;  /* opposing or noisy flows are ambiguous */

    flow_x = static_cast<float>(sum_x / resultant);
    flow_y = static_cast<float>(sum_y / resultant);
    return true;
}

static void label_wrong_route_vehicle(NvDsObjectMeta *obj_meta)
{
    if (!obj_meta)
        return;

    const char *current = obj_meta->text_params.display_text;
    if (!current || std::string(current) != "WRONG ROUTE")
    {
        if (obj_meta->text_params.display_text)
            g_free(obj_meta->text_params.display_text);
        obj_meta->text_params.display_text = g_strdup("WRONG ROUTE");
    }

    obj_meta->rect_params.border_width = 4;
    obj_meta->rect_params.border_color.red = 1.0;
    obj_meta->rect_params.border_color.green = 0.0;
    obj_meta->rect_params.border_color.blue = 0.0;
    obj_meta->rect_params.border_color.alpha = 1.0;
    obj_meta->text_params.font_params.font_color.red = 1.0;
    obj_meta->text_params.font_params.font_color.green = 1.0;
    obj_meta->text_params.font_params.font_color.blue = 1.0;
    obj_meta->text_params.font_params.font_color.alpha = 1.0;
    obj_meta->text_params.set_bg_clr = 1;
    obj_meta->text_params.text_bg_clr.red = 0.85;
    obj_meta->text_params.text_bg_clr.green = 0.0;
    obj_meta->text_params.text_bg_clr.blue = 0.0;
    obj_meta->text_params.text_bg_clr.alpha = 0.95;
}

/* apply the outcome of one save attempt (video thread only) */
static void handle_save_result(
    const SaveResult &r,
    int frame_num)
{
    auto it = track_states.find(r.track_id);
    const bool wrong_route = (r.violation_type == "wrong_route");

    if (r.success)
    {
        if (wrong_route)
            wrong_route_captured_keys.insert(r.key);
        else
            captured_keys.insert(r.key);

        consecutive_encoder_failures = 0;
        if (it != track_states.end())
        {
            if (wrong_route)
            {
                it->second.wrong_route_cap = CaptureState::Saved;
                it->second.wrong_route_waiting = false;
            }
            else
            {
                it->second.cap = CaptureState::Saved;
                it->second.crossed_waiting = false;
            }
        }
        return;
    }

    if (r.encoder_related)
    {
        consecutive_encoder_failures++;
        if (consecutive_encoder_failures % 3 == 0)
        {
            g_printerr(
                "ENCODER CHECK FAILED: %d snapshot encodes in a row failed. "
                "Check the installed JPEG/nvjpeg library and the "
                "nvds_obj_encode library you linked.\n",
                consecutive_encoder_failures);
        }
    }

    if (it == track_states.end())
        return;

    TrackState &ts = it->second;
    int attempts = 0;
    int *next_retry_frame = nullptr;
    if (wrong_route)
    {
        ts.wrong_route_cap = CaptureState::Failed;
        attempts = ++ts.wrong_route_attempts;
        next_retry_frame = &ts.wrong_route_next_retry_frame;
    }
    else
    {
        ts.cap = CaptureState::Failed;
        attempts = ++ts.attempts;
        next_retry_frame = &ts.next_retry_frame;
    }

    int shift = attempts - 1;
    if (shift > 4) shift = 4;
    int gap = SNAPSHOT_RETRY_GAP_FRAMES << shift;
    if (gap > SNAPSHOT_RETRY_MAX_GAP_FRAMES)
        gap = SNAPSHOT_RETRY_MAX_GAP_FRAMES;
    *next_retry_frame = frame_num + gap;
}

static void drain_save_results()
{
    std::vector<SaveResult> done;

    {
        std::lock_guard<std::mutex> lock(save_mutex);
        done.swap(save_results);
    }

    for (const SaveResult &r : done)
        handle_save_result(r, last_seen_frame_num);
}


static void bump_epoch(const char *reason)
{
    tracker_epoch++;
    track_states.clear();
    id_generation.clear();

    std::cout << "[TRACKER] restart detected (" << reason
              << "), epoch=" << tracker_epoch << std::endl;
}


static GstPadProbeReturn infer_src_probe(
    GstPad *,
    GstPadProbeInfo *info,
    gpointer)
{
    refresh_lane_1_signal_state();
    const bool line_enabled = crosswalk_config.detected &&
        crosswalk_config.line_y >= 0 && crosswalk_config.frame_height > 0;

    GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);

    if (!buf)
        return GST_PAD_PROBE_OK;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    if (!batch_meta)
        return GST_PAD_PROBE_OK;

    /* results of saves that finished since the last frame */
    drain_save_results();

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list;
         l_frame;
         l_frame = l_frame->next)
    {
        auto *frame_meta = static_cast<NvDsFrameMeta *>(l_frame->data);

        if (!frame_meta)
            continue;

        const int line_y = line_enabled ? compute_line_y(frame_meta) : -1;
        const int frame_num = static_cast<int>(frame_meta->frame_num);
        const guint64 pts = frame_meta->buf_pts;

        /*
         * Stream / tracker restart -> tracker ids may be reused ->
         * new epoch. Signals: the camera source was re-linked, the
         * frame counter went backwards, or the timestamp jumped back.
         */
        const gint links = g_atomic_int_get(&source_link_count);

        if (links != source_link_seen)
        {
            if (source_link_seen > 0)
                bump_epoch("camera source re-linked");

            source_link_seen = links;
        }
        else if (last_seen_frame_num >= 0 && frame_num < last_seen_frame_num)
        {
            bump_epoch("frame counter went backwards");
        }
        else if (last_seen_pts > 0 && pts + GST_SECOND < last_seen_pts)
        {
            bump_epoch("timestamp went backwards");
        }

        last_seen_frame_num = frame_num;
        last_seen_pts = pts;

        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list;
             l_obj;
             l_obj = l_obj->next)
        {
            auto *obj_meta = static_cast<NvDsObjectMeta *>(l_obj->data);

            if (!obj_meta)
                continue;

            /* tracker has not assigned an id yet */
            if (obj_meta->object_id == UNTRACKED_OBJECT_ID)
                continue;

            const std::string object_name = obj_meta->obj_label;

            if (!is_vehicle_class(object_name))
                continue;

            const guint64 id = obj_meta->object_id;

            /* bottom edge of the bounding box */
            const float y =
                obj_meta->rect_params.top +
                obj_meta->rect_params.height;

            const int zone = !line_enabled ? 0 :
                ((y <= line_y - LINE_HYSTERESIS_PX) ? -1 :
                 (y >= line_y + LINE_HYSTERESIS_PX) ?  1 : 0);

            auto it = track_states.find(id);

            /* same id back after a long gap = tracker reused it */
            if (it != track_states.end() &&
                frame_num - it->second.last_frame > ID_REUSE_GAP_FRAMES)
            {
                track_states.erase(it);
                it = track_states.end();
            }

            if (it == track_states.end())
            {
                /* first sighting (of this life of the id) */
                auto g = id_generation.find(id);

                if (g == id_generation.end())
                    id_generation[id] = 0;
                else
                    g->second++;

                TrackState ts;
                ts.last_y = y;
                ts.last_frame = frame_num;
                ts.side = zone;
                MotionPoint first_point;
                first_point.x = obj_meta->rect_params.left + obj_meta->rect_params.width * 0.5f;
                first_point.y = obj_meta->rect_params.top + obj_meta->rect_params.height * 0.5f;
                first_point.frame = frame_num;
                ts.motion_history.push_back(first_point);

                track_states[id] = ts;
                continue;
            }

            TrackState &ts = it->second;

            ts.last_y = y;
            ts.last_frame = frame_num;
            ts.lost_logged = false;
            ts.wrong_route_lost_logged = false;

            const guint32 gen = id_generation[id];
            const CaptureKey key(
                (tracker_epoch << 20) | (gen & 0xFFFFF),
                id);

            /* Update rolling centre-position history and track displacement. */
            MotionPoint point;
            point.x = obj_meta->rect_params.left + obj_meta->rect_params.width * 0.5f;
            point.y = obj_meta->rect_params.top + obj_meta->rect_params.height * 0.5f;
            point.frame = frame_num;
            ts.motion_history.push_back(point);
            while (ts.motion_history.size() > WRONG_ROUTE_HISTORY_POINTS)
                ts.motion_history.pop_front();

            if (!ts.motion_history.empty() &&
                frame_num - ts.motion_history.front().frame >= WRONG_ROUTE_MIN_HISTORY_FRAMES)
            {
                const MotionPoint &oldest = ts.motion_history.front();
                const float dx = point.x - oldest.x;
                const float dy = point.y - oldest.y;
                if (std::hypot(dx, dy) >= WRONG_ROUTE_MIN_DISPLACEMENT_PX)
                {
                    ts.motion_dx = dx;
                    ts.motion_dy = dy;
                    ts.motion_valid = true;
                    ts.motion_last_frame = frame_num;
                }
            }

            /* Dynamic dominant direction is learned from recent moving tracks. */
            float flow_x = 0.0f, flow_y = 0.0f, flow_strength = 0.0f;
            int flow_tracks = 0;
            const bool flow_valid = estimate_dominant_flow(
                frame_num, flow_x, flow_y, flow_tracks, flow_strength);

            if (!ts.wrong_route_confirmed && flow_valid && ts.motion_valid &&
                frame_num - ts.motion_last_frame <= WRONG_ROUTE_MAX_MOTION_AGE)
            {
                const float motion_mag = std::hypot(ts.motion_dx, ts.motion_dy);
                if (motion_mag > 0.0f)
                {
                    const float dot = (ts.motion_dx / motion_mag) * flow_x +
                                      (ts.motion_dy / motion_mag) * flow_y;
                    if (dot <= WRONG_ROUTE_OPPOSITE_DOT)
                        ts.opposite_count++;
                    else
                        ts.opposite_count = 0;

                    if (ts.opposite_count >= WRONG_ROUTE_CONFIRM_FRAMES)
                    {
                        ts.wrong_route_confirmed = true;
                        ts.wrong_route_waiting = true;
                        ts.wrong_route_cap = CaptureState::Idle;
                        ts.wrong_route_attempts = 0;
                        ts.wrong_route_next_retry_frame = frame_num;
                        g_print("WRONG ROUTE CONFIRMED: %s id=%llu flow_tracks=%d strength=%.2f\n",
                                object_name.c_str(),
                                static_cast<unsigned long long>(id),
                                flow_tracks, flow_strength);
                    }
                }
            }

            if (ts.wrong_route_confirmed)
                label_wrong_route_vehicle(obj_meta);

            /* ---- Existing line-crossing logic is unchanged and optional. ---- */
            if (!captured_keys.count(key) && line_enabled && zone != 0)
            {
                if (ts.side == 0)
                {
                    ts.side = zone;
                    ts.cand_side = 0;
                    ts.cand_count = 0;
                }
                else if (zone == ts.side)
                {
                    ts.cand_side = 0;
                    ts.cand_count = 0;
                }
                else
                {
                    if (ts.cand_side == zone)
                    {
                        ts.cand_count++;
                    }
                    else
                    {
                        ts.cand_side = zone;
                        ts.cand_count = 1;
                    }

                    if (ts.cand_count >= CROSS_CONFIRM_FRAMES)
                    {
                        ts.side = zone;
                        ts.cand_side = 0;
                        ts.cand_count = 0;

                        if (lane_1_signal == "RED")
                        {
                            ts.crossed_waiting = true;
                            ts.cap = CaptureState::Idle;
                            ts.attempts = 0;
                            ts.next_retry_frame = frame_num;

                            std::cout << "VEHICLE CROSSED LINE DURING RED SIGNAL: "
                                      << object_name
                                      << " id=" << id
                                      << " confidence=" << obj_meta->confidence
                                      << std::endl;
                        }
                        else
                        {
                            ts.crossed_waiting = false;
                            std::cout << "LINE CROSSING IGNORED: Lane 1 signal is "
                                      << lane_1_signal << " (not RED), id=" << id
                                      << std::endl;
                        }
                    }
                }
            }

            /* ---- Wrong-route snapshots ignore signal state and crosswalk line. ---- */
            if (ts.wrong_route_waiting &&
                !wrong_route_captured_keys.count(key) &&
                ts.wrong_route_cap != CaptureState::Pending &&
                frame_num >= ts.wrong_route_next_retry_frame &&
                !save_job_in_flight())
            {
                ts.wrong_route_cap = CaptureState::Pending;
                bool encoder_related = false;
                const bool started = start_vehicle_crop(
                    buf, frame_meta, obj_meta, object_name,
                    obj_meta->confidence, id, tracker_epoch, gen, key,
                    "wrong_route", encoder_related);
                if (!started)
                {
                    SaveResult r;
                    r.key = key;
                    r.track_id = id;
                    r.violation_type = "wrong_route";
                    r.success = false;
                    r.encoder_related = encoder_related;
                    handle_save_result(r, frame_num);
                }
            }

            /* ---- save (or retry) for a confirmed red-light crossing ---- */
            if (!captured_keys.count(key) &&
                ts.crossed_waiting &&
                lane_1_signal == "RED" &&
                ts.cap != CaptureState::Pending &&
                frame_num >= ts.next_retry_frame &&
                !save_job_in_flight())
            {
                ts.cap = CaptureState::Pending;     /* reserve */

                bool encoder_related = false;

                const bool started = start_vehicle_crop(
                    buf,
                    frame_meta,
                    obj_meta,
                    object_name,
                    obj_meta->confidence,
                    id,
                    tracker_epoch,
                    gen,
                    key,
                    "red_light_violation",
                    encoder_related);

                if (!started)
                {
                    SaveResult r;
                    r.key = key;
                    r.track_id = id;
                    r.violation_type = "red_light_violation";
                    r.success = false;
                    r.encoder_related = encoder_related;

                    handle_save_result(r, frame_num);
                }
            }
        }

        /* report vehicles that crossed but left view before a save worked */
        if (frame_num % 30 == 0)
        {
            for (auto &kv : track_states)
            {
                TrackState &t = kv.second;

                if (t.crossed_waiting &&
                    !t.lost_logged &&
                    frame_num - t.last_frame > 90)
                {
                    t.lost_logged = true;
                    g_printerr(
                        "id=%llu crossed the line but left view before a "
                        "snapshot was saved (still waiting if it returns)\n",
                        static_cast<unsigned long long>(kv.first));
                }
                if (t.wrong_route_waiting &&
                    !t.wrong_route_lost_logged &&
                    frame_num - t.last_frame > 90)
                {
                    t.wrong_route_lost_logged = true;
                    g_printerr(
                        "id=%llu was marked WRONG ROUTE but left view before "
                        "its snapshot was saved\n",
                        static_cast<unsigned long long>(kv.first));
                }
            }
        }

        /* forget old POSITIONS only; captured_keys is never erased */
        if (track_states.size() > 1000)
        {
            for (auto it = track_states.begin();
                 it != track_states.end();)
            {
                if (frame_num - it->second.last_frame > 900)
                    it = track_states.erase(it);
                else
                    ++it;
            }
        }
    }

    return GST_PAD_PROBE_OK;
}


/*
 * ============================================================
 * nvurisrcbin dynamic pad -> streammux sink_0
 * ============================================================
 */

static GstPadProbeReturn crosswalk_line_probe(
    GstPad *pad,
    GstPadProbeInfo *info,
    gpointer user_data)
{
    refresh_lane_1_signal_state();

    GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf)
        return GST_PAD_PROBE_OK;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta)
        return GST_PAD_PROBE_OK;

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list;
         l_frame != nullptr;
         l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta =
            static_cast<NvDsFrameMeta *>(l_frame->data);

        if (!frame_meta)
            continue;

        /*
         * Display filtering only: keep all object metadata available to
         * inference, tracking, line-crossing and snapshot logic, but hide
         * OSD boxes and labels for non-vehicle classes in the AI RTSP output.
         */
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list;
             l_obj != nullptr;
             l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj_meta =
                static_cast<NvDsObjectMeta *>(l_obj->data);
            if (!obj_meta)
                continue;

            const std::string object_name = obj_meta->obj_label;
            if (!is_vehicle_class(object_name))
            {
                obj_meta->rect_params.border_width = 0;
                if (obj_meta->text_params.display_text)
                    obj_meta->text_params.display_text[0] = '\0';
                obj_meta->text_params.set_bg_clr = 0;
            }
        }

        int frame_width = (frame_meta->pipeline_width > 0)
            ? static_cast<int>(frame_meta->pipeline_width)
            : MUX_WIDTH;

        if (crosswalk_config.detected)
        {
            int line_y = compute_line_y(frame_meta);

            NvDsDisplayMeta *line_meta =
                nvds_acquire_display_meta_from_pool(batch_meta);

            if (line_meta)
            {
                line_meta->num_lines = 1;
                NvOSD_LineParams &line = line_meta->line_params[0];
                line.x1 = 0;
                line.y1 = line_y;
                line.x2 = frame_width;
                line.y2 = line_y;
                line.line_width = 5;
                line.line_color.red = 1.0;
                line.line_color.green = 0.0;
                line.line_color.blue = 0.0;
                line.line_color.alpha = 1.0;
                nvds_add_display_meta_to_frame(frame_meta, line_meta);
            }
        }

        /* Lane 1 signal status overlay on the AI-processed stream. */
        NvDsDisplayMeta *signal_meta =
            nvds_acquire_display_meta_from_pool(batch_meta);

        if (signal_meta)
        {
            signal_meta->num_labels = 1;
            NvOSD_TextParams &label = signal_meta->text_params[0];
            const std::string label_text = "LANE 1: " + lane_1_signal;
            label.display_text = g_strdup(label_text.c_str());
            label.x_offset = 24;
            label.y_offset = 24;
            label.font_params.font_name = const_cast<gchar *>("Serif");
            label.font_params.font_size = 24;
            label.font_params.font_color.red = 1.0;
            label.font_params.font_color.green = 1.0;
            label.font_params.font_color.blue = 1.0;
            label.font_params.font_color.alpha = 1.0;
            label.set_bg_clr = 1;
            label.text_bg_clr.red = (lane_1_signal == "RED") ? 0.75 : 0.12;
            label.text_bg_clr.green = (lane_1_signal == "GREEN") ? 0.55 : 0.08;
            label.text_bg_clr.blue = 0.08;
            label.text_bg_clr.alpha = 0.95;
            nvds_add_display_meta_to_frame(frame_meta, signal_meta);
        }
    }

    return GST_PAD_PROBE_OK;
}


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
        g_atomic_int_inc(&source_link_count);

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
    bool crosswalk_loaded = load_crosswalk_line_config();

    if (!crosswalk_loaded)
    {
        std::cerr << "[CROSSWALK] WARNING: fixed line will not be drawn" << std::endl;
    }
    else
    {
        std::cout << "[CROSSWALK] Fixed line enabled at Y="
                  << crosswalk_config.line_y << std::endl;
    }

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

    obj_enc_ctx = OBJ_ENC_CREATE_CONTEXT();

    if (!obj_enc_ctx)
    {
        std::cerr
            << "ERROR: nvds_obj_enc context could not be created. "
               "Vehicle snapshots are DISABLED (stream keeps running)."
            << std::endl;
    }


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

    GstElement *tracker =
        gst_element_factory_make(
            "nvtracker",
            "vehicle-tracker");
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
        !tracker ||
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
        G_OBJECT(tracker),
        "ll-lib-file",
        TRACKER_LIB,
        "ll-config-file",
        TRACKER_CONFIG,
        "tracker-width",
        TRACKER_WIDTH,
        "tracker-height",
        TRACKER_HEIGHT,
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
            "width=1920,"
            "height=1080");

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
        tracker,
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

    if (!gst_element_link_many(
            streammux,
            infer,
            tracker,
            osd,
            tee,
            nullptr))
    {
        std::cerr
            << "ERROR: Failed to link "
               "streammux -> nvinfer -> nvtracker -> "
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


    /* --- Fixed Crosswalk line probe --- */
    GstPad *osd_sink = gst_element_get_static_pad(osd, "sink");
    if (!osd_sink)
    {
        std::cerr << "[CROSSWALK] Failed to get OSD sink pad" << std::endl;
    }
    else
    {
        gst_pad_add_probe(
            osd_sink,
            GST_PAD_PROBE_TYPE_BUFFER,
            crosswalk_line_probe,
            nullptr,
            nullptr);

        gst_object_unref(osd_sink);
        std::cout << "[CROSSWALK] Line probe attached to OSD sink" << std::endl;
    }


    /* --- Detection probe --- */

    /*
     * OSD *sink* pad: the frame is still clean here (boxes, labels and
     * the red line are drawn afterwards), and it is the exact frame
     * the crossing was confirmed on.
     */
    GstPad *osd_src =
        gst_element_get_static_pad(
            osd,
            "sink");

    if (!osd_src)
    {
        std::cerr
            << "ERROR: Failed to get "
               "shared nvdsosd sink pad."
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
        " Vehicle snapshots: %s (one jpg per vehicle crossing the line)\n",
        VEHICLE_SNAPSHOT_DIR);

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

    wait_for_save_jobs();

    if (obj_enc_ctx)
        nvds_obj_enc_destroy_context(obj_enc_ctx);

    gst_object_unref(pipeline);

    g_main_loop_unref(main_loop);

    std::cout
        << "Pipeline stopped."
        << std::endl;

    return 0;
}
