#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <regex>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <iomanip>
#include <cctype>
#include <cmath>
#include <chrono>
#include <thread>

namespace fs = std::filesystem;


/*
 * ============================================================
 * Configuration
 * ============================================================
 */

static const std::string RECORDINGS_ROOT =
    "/root/video_recorder/recordings";

static const std::string OUTPUT_DIR =
    "/root/video_recorder/events/videos";

static const int BEFORE_SECONDS = 60;
static const int AFTER_SECONDS  = 60;

static const int TARGET_EVENT_SECONDS = 120;

/*
 * Minimum usable footage required to generate an event.
 *
 * 120 sec = perfect
 * 119 sec = generate
 * 117 sec = generate
 * 115 sec = generate
 * 110 sec = generate + warning
 * <110 sec = reject
 */
static const int MIN_EVENT_SECONDS = 110;

/*
 * Small gaps between recording files are accepted.
 *
 * Example:
 *
 * File 1 ends: 06:13:49
 * File 2 starts: 06:13:51
 *
 * Gap = 2 seconds
 *
 * This is acceptable.
 */
static const int MAX_RECORDING_GAP_SECONDS = 8;


/*
 * ============================================================
 * Recording information
 * ============================================================
 */

struct VideoFile
{
    std::string path;

    std::time_t start_time = 0;
    double duration = 0.0;

    std::time_t end_time() const
    {
        return start_time +
               static_cast<std::time_t>(duration);
    }
};


/*
 * ============================================================
 * Execute command and capture output
 * ============================================================
 */

static std::string
run_command(
    const std::string &command)
{
    std::array<char, 512> buffer;

    std::string result;

    FILE *pipe =
        popen(command.c_str(), "r");

    if (!pipe)
    {
        return "";
    }

    while (fgets(
        buffer.data(),
        buffer.size(),
        pipe))
    {
        result += buffer.data();
    }

    pclose(pipe);

    return result;
}


/*
 * ============================================================
 * Convert tm to time_t
 * ============================================================
 */

static std::time_t
make_time_utc(
    std::tm tm_value)
{
    /*
     * Recording timestamps are UTC.
     */

    return timegm(&tm_value);
}

/*
 * ============================================================
 * Parse timestamp:
 *
 * YYYY-MM-DDTHH:MM:SS
 *
 * Example:
 *
 * 2026-10-05T10:37:15
 * ============================================================
 */

static bool
parse_event_timestamp(
    const std::string &timestamp,
    std::time_t &result)
{
    std::tm tm_value{};

    if (timestamp.size() != 19)
    {
        return false;
    }

    if (timestamp[4] != '-' ||
        timestamp[7] != '-' ||
        timestamp[10] != 'T' ||
        timestamp[13] != ':' ||
        timestamp[16] != ':')
    {
        return false;
    }

    try
    {
        tm_value.tm_year =
            std::stoi(timestamp.substr(0, 4)) - 1900;

        tm_value.tm_mon =
            std::stoi(timestamp.substr(5, 2)) - 1;

        tm_value.tm_mday =
            std::stoi(timestamp.substr(8, 2));

        tm_value.tm_hour =
            std::stoi(timestamp.substr(11, 2));

        tm_value.tm_min =
            std::stoi(timestamp.substr(14, 2));

        tm_value.tm_sec =
            std::stoi(timestamp.substr(17, 2));

        tm_value.tm_isdst = 0;
    }
    catch (...)
    {
        return false;
    }

    result = make_time_utc(tm_value);

    return true;
}

/*
 * ============================================================
 * Format time
 * ============================================================
 */

static std::string
format_time(
    std::time_t value)
{
    std::tm *tm_value =
        gmtime(&value);

    if (!tm_value)
    {
        return "";
    }

    char buffer[64];

    strftime(
        buffer,
        sizeof(buffer),
        "%Y-%m-%d %H:%M:%S",
        tm_value);

    return buffer;
}


/*
 * ============================================================
 * Format filename timestamp
 * ============================================================
 */

static std::string
format_filename_timestamp(
    std::time_t value)
{
    std::tm *tm_value =
        gmtime(&value);

    if (!tm_value)
    {
        return "";
    }

    char buffer[64];

    strftime(
        buffer,
        sizeof(buffer),
        "%Y%m%d_%H%M%S",
        tm_value);

    return buffer;
}


/*
 * ============================================================
 * Safe filename
 * ============================================================
 */

static std::string
safe_filename(
    std::string value)
{
    for (char &c : value)
    {
        if (!std::isalnum(
                static_cast<unsigned char>(c)) &&
            c != '_' &&
            c != '-')
        {
            c = '_';
        }
    }

    return value;
}


/*
 * ============================================================
 * Read JSON string value
 * ============================================================
 */

static bool
get_json_string(
    const std::string &json,
    const std::string &key,
    std::string &value)
{
    std::string pattern =
        "\"" + key +
        "\"\\s*:\\s*\"([^\"]*)\"";

    std::regex expression(pattern);

    std::smatch match;

    if (!std::regex_search(
            json,
            match,
            expression))
    {
        return false;
    }

    value = match[1];

    return true;
}


/*
 * ============================================================
 * Read detection JSON
 * ============================================================
 */

static bool
read_detection_json(
    const std::string &json_path,
    std::string &object_name,
    std::string &timestamp)
{
    std::ifstream file(json_path);

    if (!file)
    {
        std::cerr
            << "ERROR: Cannot open JSON: "
            << json_path
            << std::endl;

        return false;
    }

    std::stringstream buffer;

    buffer << file.rdbuf();

    std::string json =
        buffer.str();

    if (!get_json_string(
            json,
            "object",
            object_name))
    {
        std::cerr
            << "ERROR: object field not found"
            << std::endl;

        return false;
    }

    if (!get_json_string(
            json,
            "timestamp",
            timestamp))
    {
        std::cerr
            << "ERROR: timestamp field not found"
            << std::endl;

        return false;
    }

    return true;
}


/*
 * ============================================================
 * Get video creation time using ffprobe
 * ============================================================
 */

static bool
get_video_info(
    const std::string &path,
    std::time_t &start_time,
    double &duration)
{
    std::string command =
        "ffprobe -v error "
        "-show_entries "
        "format=duration:format_tags=creation_time "
        "-of default=noprint_wrappers=1 "
        "\"" + path + "\" 2>/dev/null";

    std::string output =
        run_command(command);

    std::string creation_time;

    std::stringstream ss(output);

    std::string line;

    duration = 0.0;

    while (std::getline(
        ss,
        line))
    {
        if (line.rfind(
                "duration=",
                0) == 0)
        {
            try
            {
                duration =
                    std::stod(
                        line.substr(9));
            }
            catch (...)
            {
                return false;
            }
        }

        if (line.rfind(
                "TAG:creation_time=",
                0) == 0)
        {
            creation_time =
                line.substr(
                    std::string(
                        "TAG:creation_time=")
                        .size());
        }
    }

    /*
     * Prefer creation_time.
     */

    if (!creation_time.empty())
    {
        std::tm tm_value{};

        int year;
        int month;
        int day;
        int hour;
        int minute;
        int second;

        if (sscanf(
                creation_time.c_str(),
                "%d-%d-%dT%d:%d:%d",
                &year,
                &month,
                &day,
                &hour,
                &minute,
                &second) == 6)
        {
            tm_value.tm_year =
                year - 1900;

            tm_value.tm_mon =
                month - 1;

            tm_value.tm_mday =
                day;

            tm_value.tm_hour =
                hour;

            tm_value.tm_min =
                minute;

            tm_value.tm_sec =
                second;

            tm_value.tm_isdst = 0;

            start_time =
                make_time_utc(
                    tm_value);

            return duration > 0.0;
        }
    }

    return false;
}


/*
 * ============================================================
 * Find all recording files
 * ============================================================
 */

static std::vector<VideoFile>
find_recordings(
    std::time_t required_start,
    std::time_t required_end)
{
    std::vector<VideoFile> videos;

    if (!fs::exists(
            RECORDINGS_ROOT))
    {
        std::cerr
            << "ERROR: Recording directory does not exist: "
            << RECORDINGS_ROOT
            << std::endl;

        return videos;
    }

    for (const auto &entry :
         fs::recursive_directory_iterator(
             RECORDINGS_ROOT))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        if (entry.path().extension()
                != ".mp4")
        {
            continue;
        }

        VideoFile video;

        video.path =
            entry.path().string();

        std::cout
            << "Checking recording: "
            << video.path
            << std::endl;

        if (!get_video_info(
                video.path,
                video.start_time,
                video.duration))
        {
            std::cerr
                << "  Could not read video metadata"
                << std::endl;

            continue;
        }

        std::cout
            << "  Start: "
            << format_time(
                video.start_time)
            << std::endl;

        std::cout
            << "  Duration: "
            << std::fixed
            << std::setprecision(3)
            << video.duration
            << " sec"
            << std::endl;

        std::cout
            << "  End: "
            << format_time(
                video.end_time())
            << std::endl;

        /*
         * Select every recording that overlaps
         * the requested event period.
         */

        if (video.start_time < required_end &&
            video.end_time() > required_start)
        {
            videos.push_back(video);

            std::cout
                << "  >>> SELECTED"
                << std::endl;
        }
    }

    std::sort(
        videos.begin(),
        videos.end(),
        [](const VideoFile &a,
           const VideoFile &b)
        {
            return a.start_time <
                   b.start_time;
        });

    return videos;
}


/*
 * ============================================================
 * Calculate usable event coverage
 *
 * Allows small recording gaps up to 8 seconds.
 * ============================================================
 */

static double
calculate_available_coverage(
    const std::vector<VideoFile> &videos,
    std::time_t required_start,
    std::time_t required_end,
    std::time_t &coverage_start,
    std::time_t &coverage_end)
{
    bool started = false;

    coverage_start = 0;
    coverage_end = 0;

    for (const auto &video : videos)
    {
        std::time_t video_start =
            video.start_time;

        std::time_t video_end =
            video.end_time();

        /*
         * Ignore recordings completely before
         * requested interval.
         */

        if (video_end <= required_start)
        {
            continue;
        }

        /*
         * Ignore recordings completely after
         * requested interval.
         */

        if (video_start >= required_end)
        {
            continue;
        }

        /*
         * First usable recording.
         */

        if (!started)
        {
            coverage_start =
                std::max(
                    required_start,
                    video_start);

            coverage_end =
                std::min(
                    required_end,
                    video_end);

            started = true;

            continue;
        }

        /*
         * Calculate gap.
         */

        double gap =
            difftime(
                video_start,
                coverage_end);

        /*
         * If there is a small gap, accept it.
         *
         * The gap is NOT counted as video.
         */

        if (gap > 0)
        {
            if (gap >
                MAX_RECORDING_GAP_SECONDS)
            {
                std::cout
                    << "  Recording gap too large: "
                    << gap
                    << " sec"
                    << std::endl;

                break;
            }

            std::cout
                << "  Small recording gap accepted: "
                << gap
                << " sec"
                << std::endl;
        }

        /*
         * Extend coverage.
         */

        if (video_end > coverage_end)
        {
            coverage_end =
                std::min(
                    required_end,
                    video_end);
        }

        if (coverage_end >= required_end)
        {
            break;
        }
    }

    if (!started ||
        coverage_end <= coverage_start)
    {
        return 0.0;
    }

    return difftime(
        coverage_end,
        coverage_start);
}


/*
 * ============================================================
 * Create temporary directory
 * ============================================================
 */

static std::string
create_temp_directory()
{
    std::string command =
        "mktemp -d /tmp/video_event_XXXXXX";

    std::string result =
        run_command(command);

    while (!result.empty() &&
           (result.back() == '\n' ||
            result.back() == '\r'))
    {
        result.pop_back();
    }

    return result;
}


/*
 * ============================================================
 * Extract one segment
 *
 * IMPORTANT:
 *
 * No hardware encoding.
 *
 * Source video is already H.264.
 * We use stream copy.
 * ============================================================
 */

static bool
extract_segment(
    const VideoFile &video,
    std::time_t required_start,
    std::time_t required_end,
    const std::string &output)
{
    std::time_t segment_start =
        std::max(
            required_start,
            video.start_time);

    std::time_t segment_end =
        std::min(
            required_end,
            video.end_time());

    if (segment_end <= segment_start)
    {
        return false;
    }

    double offset =
        difftime(
            segment_start,
            video.start_time);

    double length =
        difftime(
            segment_end,
            segment_start);

    std::cout
        << "Extracting: "
        << format_time(segment_start)
        << " -> "
        << format_time(segment_end)
        << std::endl;

    std::cout
        << "  Offset: "
        << std::fixed
        << std::setprecision(3)
        << offset
        << " sec"
        << std::endl;

    std::cout
        << "  Length: "
        << length
        << " sec"
        << std::endl;

    /*
     * Stream copy.
     *
     * No h264_v4l2m2m.
     * No h264_nvmpi.
     * No libx264.
     */

    std::ostringstream command;

    command
        << "ffmpeg -y "
        << "-hide_banner "
        << "-loglevel warning "
        << "-ss "
        << std::fixed
        << std::setprecision(3)
        << offset
        << " "
        << "-i \""
        << video.path
        << "\" "
        << "-t "
        << length
        << " "
        << "-c:v copy "
        << "-c:a copy "
        << "-avoid_negative_ts make_zero "
        << "\""
        << output
        << "\"";

    std::cout
        << "Running FFmpeg stream copy..."
        << std::endl;

    int result =
        std::system(
            command.str().c_str());

    if (result != 0)
    {
        std::cerr
            << "ERROR: FFmpeg extraction failed."
            << std::endl;

        return false;
    }

    if (!fs::exists(output))
    {
        std::cerr
            << "ERROR: Output segment was not created."
            << std::endl;

        return false;
    }

    if (fs::file_size(output) == 0)
    {
        std::cerr
            << "ERROR: Output segment is empty."
            << std::endl;

        fs::remove(output);

        return false;
    }

    std::cout
        << "Segment created successfully."
        << std::endl;

    return true;
}


/*
 * ============================================================
 * Join extracted segments
 * ============================================================
 */

static bool
join_segments(
    const std::vector<std::string> &segments,
    const std::string &output)
{
    if (segments.empty())
    {
        return false;
    }

    fs::path first =
        segments.front();

    fs::path list_path =
        first.parent_path()
        / "concat.txt";

    std::ofstream list(
        list_path);

    if (!list)
    {
        std::cerr
            << "ERROR: Cannot create concat list."
            << std::endl;

        return false;
    }

    for (const auto &segment :
         segments)
    {
        /*
         * Escape single quotes if necessary.
         */

        std::string escaped =
            segment;

        std::string::size_type pos = 0;

        while ((pos =
                escaped.find(
                    "'",
                    pos))
               != std::string::npos)
        {
            escaped.replace(
                pos,
                1,
                "'\\''");

            pos += 4;
        }

        list
            << "file '"
            << escaped
            << "'\n";
    }

    list.close();

    std::ostringstream command;

    command
        << "ffmpeg -y "
        << "-hide_banner "
        << "-loglevel warning "
        << "-f concat "
        << "-safe 0 "
        << "-i \""
        << list_path.string()
        << "\" "
        << "-c copy "
        << "-avoid_negative_ts make_zero "
        << "\""
        << output
        << "\"";

    int result =
        std::system(
            command.str().c_str());

    fs::remove(
        list_path);

    if (result != 0)
    {
        return false;
    }

    if (!fs::exists(output))
    {
        return false;
    }

    if (fs::file_size(output) == 0)
    {
        return false;
    }

    return true;
}


/*
 * ============================================================
 * Get final output duration
 * ============================================================
 */

static double
get_video_duration(
    const std::string &path)
{
    std::string command =
        "ffprobe -v error "
        "-show_entries format=duration "
        "-of default=noprint_wrappers=1:nokey=1 "
        "\"" + path + "\" 2>/dev/null";

    std::string result =
        run_command(command);

    while (!result.empty() &&
           (result.back() == '\n' ||
            result.back() == '\r'))
    {
        result.pop_back();
    }

    if (result.empty())
    {
        return 0.0;
    }

    try
    {
        return std::stod(result);
    }
    catch (...)
    {
        return 0.0;
    }
}


/*
 * ============================================================
 * Event processing
 * ============================================================
 */

static bool process_event(const std::string &json_path)
{
    std::string object_name;
    std::string timestamp;

    if (!read_detection_json(json_path, object_name, timestamp))
    {
        std::cerr << "ERROR: Cannot read detection JSON: " << json_path << std::endl;
        return false;
    }

    std::cout << "\n============================================================" << std::endl;
    std::cout << "Processing event JSON" << std::endl;
    std::cout << "JSON: " << json_path << std::endl;
    std::cout << "Object: " << object_name << std::endl;
    std::cout << "Timestamp: " << timestamp << std::endl;

    std::time_t detection_time;
    if (!parse_event_timestamp(timestamp, detection_time))
    {
        std::cerr << "ERROR: Invalid timestamp: " << timestamp << std::endl;
        return false;
    }

    std::time_t required_start = detection_time - BEFORE_SECONDS;
    std::time_t required_end   = detection_time + AFTER_SECONDS;

    std::cout << "Required event video:" << std::endl;
    std::cout << "  Start:     " << format_time(required_start) << std::endl;
    std::cout << "  Detection: " << format_time(detection_time) << std::endl;
    std::cout << "  End:       " << format_time(required_end) << std::endl;
    std::cout << "  Target:    " << TARGET_EVENT_SECONDS << " seconds" << std::endl;
    std::cout << "  Minimum:   " << MIN_EVENT_SECONDS << " seconds" << std::endl;

    std::error_code ec;
    fs::create_directories(OUTPUT_DIR, ec);
    if (ec)
    {
        std::cerr << "ERROR: Cannot create output directory: " << OUTPUT_DIR << std::endl;
        return false;
    }

    std::string output_filename =
        safe_filename(object_name) + "_" +
        format_filename_timestamp(detection_time) + ".mp4";

    std::string output_path = OUTPUT_DIR + "/" + output_filename;

    // Persistent duplicate protection. If the event video already exists,
    // this JSON has already been successfully processed.
    if (fs::exists(output_path) && fs::file_size(output_path) > 0)
    {
        std::cout << "Event video already exists. Skipping: " << output_path << std::endl;
        return true;
    }

    std::cout << "\nSearching recordings..." << std::endl;

    std::vector<VideoFile> videos =
        find_recordings(required_start, required_end);

    if (videos.empty())
    {
        std::cout << "No recordings currently overlap the event period. Will retry." << std::endl;
        return false;
    }

    std::cout << "\nSelected " << videos.size() << " recording file(s)." << std::endl;

    std::time_t coverage_start = 0;
    std::time_t coverage_end = 0;

    double available_seconds =
        calculate_available_coverage(
            videos,
            required_start,
            required_end,
            coverage_start,
            coverage_end);

    std::cout << "\nAvailable event footage: "
              << std::fixed << std::setprecision(3)
              << available_seconds << " seconds" << std::endl;

    if (coverage_start != 0)
    {
        std::cout << "Available start: " << format_time(coverage_start) << std::endl;
        std::cout << "Available end:   " << format_time(coverage_end) << std::endl;
    }

    if (available_seconds < MIN_EVENT_SECONDS)
    {
        // Do not immediately reject a newly-created event. The recorder may
        // still be writing the +60 seconds after detection.
        std::time_t now = std::time(nullptr);
        const std::time_t retry_deadline = required_end + MAX_RECORDING_GAP_SECONDS + 5;

        if (now < retry_deadline)
        {
            std::cout << "\nEVENT WAITING" << std::endl;
            std::cout << "Available footage: " << available_seconds << " seconds" << std::endl;
            std::cout << "Minimum required: " << MIN_EVENT_SECONDS << " seconds" << std::endl;
            std::cout << "Recording may still be growing. Will retry automatically." << std::endl;
            std::cout << "============================================================" << std::endl;
            return false;
        }

        std::cerr << "\nEVENT VIDEO REJECTED" << std::endl;
        std::cerr << "Available footage: " << available_seconds << " seconds" << std::endl;
        std::cerr << "Minimum required: " << MIN_EVENT_SECONDS << " seconds" << std::endl;
        std::cerr << "Retry window expired; not enough footage." << std::endl;
        std::cerr << "============================================================" << std::endl;
        return true;
    }

    if (available_seconds < TARGET_EVENT_SECONDS)
    {
        std::cout << "\nWARNING: Event video is shorter than 2 minutes." << std::endl;
        std::cout << "Available footage: " << available_seconds << " seconds" << std::endl;
        std::cout << "Target duration: " << TARGET_EVENT_SECONDS << " seconds" << std::endl;
    }
    else
    {
        std::cout << "\nFull 2-minute coverage available." << std::endl;
    }

    std::cout << "============================================================" << std::endl;

    std::string temp_dir = create_temp_directory();
    if (temp_dir.empty())
    {
        std::cerr << "ERROR: Could not create temporary directory." << std::endl;
        return false;
    }

    std::cout << "\nTemporary directory: " << temp_dir << std::endl;

    std::vector<std::string> segments;
    int segment_number = 0;

    // Only use footage that is actually available. This produces 120 sec
    // when complete, or 110-119 sec when a small amount is missing.
    std::time_t usable_end = std::min(required_end, coverage_end);

    for (const auto &video : videos)
    {
        std::time_t segment_start =
            std::max(required_start, video.start_time);

        std::time_t segment_end =
            std::min(usable_end, video.end_time());

        if (segment_end <= segment_start)
            continue;

        std::ostringstream filename;
        filename << temp_dir << "/segment_"
                 << std::setfill('0') << std::setw(4)
                 << segment_number++ << ".mp4";

        if (extract_segment(
                video,
                segment_start,
                segment_end,
                filename.str()))
        {
            segments.push_back(filename.str());
        }
        else
        {
            std::cerr << "WARNING: Failed to extract segment from "
                      << video.path << std::endl;
        }
    }

    if (segments.empty())
    {
        std::cerr << "ERROR: No video segments were created. Will retry." << std::endl;
        fs::remove_all(temp_dir);
        return false;
    }

    std::cout << "\nJoining segments..." << std::endl;

    if (!join_segments(segments, output_path))
    {
        std::cerr << "ERROR: Failed to join video segments. Will retry." << std::endl;
        fs::remove_all(temp_dir);
        return false;
    }

    double final_duration = get_video_duration(output_path);

    std::cout << "\nFinal generated video duration: "
              << std::fixed << std::setprecision(3)
              << final_duration << " seconds" << std::endl;

    fs::remove_all(temp_dir);

    if (final_duration < MIN_EVENT_SECONDS)
    {
        std::cerr << "ERROR: Final video is below minimum duration. Removing it." << std::endl;
        std::error_code remove_ec;
        fs::remove(output_path, remove_ec);
        return false;
    }

    std::cout << "\n============================================================" << std::endl;
    std::cout << "EVENT VIDEO CREATED" << std::endl;
    std::cout << "Object: " << object_name << std::endl;
    std::cout << "Detection: " << format_time(detection_time) << std::endl;
    std::cout << "Output: " << output_path << std::endl;
    std::cout << "Duration: " << final_duration << " seconds" << std::endl;
    std::cout << "============================================================" << std::endl;

    return true;
}


/*
 * ============================================================
 * Find pending JSON files
 * ============================================================
 */

static std::vector<std::string> find_pending_json_files()
{
    const std::string PENDING_DIR =
        "/root/video_recorder/events/pending";

    std::vector<std::string> files;

    if (!fs::exists(PENDING_DIR))
    {
        std::error_code ec;
        fs::create_directories(PENDING_DIR, ec);
        if (ec)
        {
            std::cerr << "ERROR: Cannot create pending directory: "
                      << PENDING_DIR << std::endl;
        }
        return files;
    }

    std::error_code ec;
    for (const auto &entry : fs::directory_iterator(PENDING_DIR, ec))
    {
        if (ec)
            break;

        if (!entry.is_regular_file())
            continue;

        if (entry.path().extension() == ".json")
            files.push_back(entry.path().string());
    }

    std::sort(files.begin(), files.end());
    return files;
}


/*
 * ============================================================
 * Main - automatic event watcher
 * ============================================================
 */

int main()
{
    const std::string PENDING_DIR =
        "/root/video_recorder/events/pending";

    std::cout
        << "============================================================\n"
        << " Radian Automatic 2-Minute Event Video Generator\n"
        << "============================================================\n"
        << "Pending directory: " << PENDING_DIR << "\n"
        << "Output directory:  " << OUTPUT_DIR << "\n"
        << "Scan interval:     2 seconds\n"
        << "============================================================\n";

    std::cout << "Checking ffprobe..." << std::endl;
    if (std::system("command -v ffprobe >/dev/null 2>&1") != 0)
    {
        std::cerr << "ERROR: ffprobe is not installed." << std::endl;
        return 1;
    }

    std::cout << "Checking ffmpeg..." << std::endl;
    if (std::system("command -v ffmpeg >/dev/null 2>&1") != 0)
    {
        std::cerr << "ERROR: ffmpeg is not installed." << std::endl;
        return 1;
    }

    std::error_code ec;
    fs::create_directories(PENDING_DIR, ec);
    if (ec)
    {
        std::cerr << "ERROR: Cannot create pending directory: "
                  << PENDING_DIR << std::endl;
        return 1;
    }

    fs::create_directories(OUTPUT_DIR, ec);
    if (ec)
    {
        std::cerr << "ERROR: Cannot create output directory: "
                  << OUTPUT_DIR << std::endl;
        return 1;
    }

    std::cout << "\nAutomatic watcher started." << std::endl;
    std::cout << "Scanning for detection JSON files continuously..." << std::endl;

    while (true)
    {
        std::vector<std::string> pending_files = find_pending_json_files();

        if (pending_files.empty())
        {
            std::cout << "." << std::flush;
        }
        else
        {
            std::cout << "\nFound " << pending_files.size()
                      << " pending JSON file(s)." << std::endl;

            for (const auto &json_path : pending_files)
            {
                // A successfully-created output file is the persistent
                // completion marker. process_event() skips it automatically.
                std::cout << "\n[EVENT] " << json_path << std::endl;

                bool completed = process_event(json_path);

                if (completed)
                {
                    // The JSON is intentionally NOT deleted. It remains as
                    // the detection/event record for later use.
                    std::cout << "[EVENT] Processing completed or permanently rejected."
                              << std::endl;
                }
                else
                {
                    // Not enough footage yet, FFmpeg failed, or recordings
                    // are still being written. It remains in pending/ and
                    // will automatically be checked again.
                    std::cout << "[EVENT] Still pending. Will retry automatically." << std::endl;
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
}
