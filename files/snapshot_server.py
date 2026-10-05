
#!/usr/bin/env python3

from flask import Flask, jsonify, send_from_directory
import os
import time

# ============================================================
# Configuration
# ============================================================

HOST = "0.0.0.0"
PORT = 5000

BASE_DIR = "/root/video_recorder/snapshots"

SNAPSHOT_DIRS = {
    "people": os.path.join(BASE_DIR, "people"),
    "vehicles": os.path.join(BASE_DIR, "vehicles"),
    "animals": os.path.join(BASE_DIR, "animals"),
    "electronics": os.path.join(BASE_DIR, "electronics"),
    "other": os.path.join(BASE_DIR, "other"),
}

ALLOWED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png"
)

# Maximum number of images returned per category
# Set to None if you want all images.
MAX_IMAGES_PER_CATEGORY = 100


# ============================================================
# Flask Application
# ============================================================

app = Flask(__name__)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

# Allow dashboard running on another machine/IP


# ============================================================
# Helper Functions
# ============================================================

def is_image(filename):
    """
    Check whether the file is an allowed image.
    """
    return filename.lower().endswith(ALLOWED_EXTENSIONS)


def get_images(category, folder):
    """
    Get images from a snapshot folder.

    Returns newest images first.
    """

    images = []

    if not os.path.isdir(folder):
        print(f"[WARNING] Snapshot folder does not exist: {folder}")
        return images

    try:
        for filename in os.listdir(folder):

            if not is_image(filename):
                continue

            filepath = os.path.join(folder, filename)

            # Make sure it is actually a file
            if not os.path.isfile(filepath):
                continue

            try:
                timestamp = os.path.getmtime(filepath)

                images.append({
                    "filename": filename,
                    "category": category,
                    "timestamp": timestamp,
                    "url": f"/snapshots/{category}/{filename}"
                })

            except OSError as error:
                print(
                    f"[WARNING] Cannot read file information "
                    f"for {filepath}: {error}"
                )

    except OSError as error:
        print(
            f"[ERROR] Cannot read snapshot directory "
            f"{folder}: {error}"
        )

    # Newest first
    images.sort(
        key=lambda item: item["timestamp"],
        reverse=True
    )

    if MAX_IMAGES_PER_CATEGORY is not None:
        images = images[:MAX_IMAGES_PER_CATEGORY]

    return images


def get_all_snapshots():
    """
    Build the complete snapshot response.

    Dashboard categories:

        People
        Vehicles
        Others

    Others contains:

        animals
        electronics
        other
    """

    result = {
        "people": [],
        "vehicles": [],
        "others": []
    }

    # --------------------------------------------------------
    # PEOPLE
    # --------------------------------------------------------

    people = get_images(
        "people",
        SNAPSHOT_DIRS["people"]
    )

    result["people"].extend(people)

    # --------------------------------------------------------
    # VEHICLES
    # --------------------------------------------------------

    vehicles = get_images(
        "vehicles",
        SNAPSHOT_DIRS["vehicles"]
    )

    result["vehicles"].extend(vehicles)

    # --------------------------------------------------------
    # OTHERS
    #
    # Animals + Electronics + Other
    # --------------------------------------------------------

    for category in (
        "animals",
        "electronics",
        "other"
    ):

        images = get_images(
            category,
            SNAPSHOT_DIRS[category]
        )

        result["others"].extend(images)

    # --------------------------------------------------------
    # Sort combined Others category
    # --------------------------------------------------------

    result["others"].sort(
        key=lambda item: item["timestamp"],
        reverse=True
    )

    if MAX_IMAGES_PER_CATEGORY is not None:
        result["others"] = result["others"][
            :MAX_IMAGES_PER_CATEGORY
        ]

    return result


# ============================================================
# API Routes
# ============================================================

@app.route("/")
def index():
    """
    Server information.
    """

    return jsonify({
        "service": "Jetson AI Snapshot Server",
        "status": "running",
        "version": "1.0",
        "api": {
            "snapshots": "/api/snapshots",
            "health": "/api/health"
        }
    })


# ------------------------------------------------------------
# Health Check
# ------------------------------------------------------------

@app.route("/api/health")
def health():
    """
    Health-check endpoint.
    """

    return jsonify({
        "status": "ok",
        "service": "snapshot_server",
        "timestamp": time.time()
    })


# ------------------------------------------------------------
# Get All Snapshots
# ------------------------------------------------------------

@app.route("/api/snapshots")
def snapshots():
    """
    Return all snapshots grouped into:

        people
        vehicles
        others
    """

    result = get_all_snapshots()

    return jsonify(result)


# ------------------------------------------------------------
# Get Snapshot Counts
# ------------------------------------------------------------

@app.route("/api/snapshots/count")
def snapshot_counts():
    """
    Return number of snapshots in each dashboard category.
    """

    data = get_all_snapshots()

    return jsonify({
        "people": len(data["people"]),
        "vehicles": len(data["vehicles"]),
        "others": len(data["others"]),
        "total": (
            len(data["people"]) +
            len(data["vehicles"]) +
            len(data["others"])
        )
    })


# ============================================================
# Image Serving
# ============================================================

@app.route(
    "/snapshots/<category>/<path:filename>"
)
def serve_snapshot(category, filename):
    """
    Serve an individual snapshot image.

    Example:

        /snapshots/people/person.jpg

        /snapshots/vehicles/car.jpg

        /snapshots/animals/dog.jpg

        /snapshots/electronics/tv.jpg

        /snapshots/other/object.jpg
    """

    if category not in SNAPSHOT_DIRS:
        return jsonify({
            "error": "Invalid snapshot category"
        }), 404

    folder = SNAPSHOT_DIRS[category]

    if not os.path.isdir(folder):
        return jsonify({
            "error": "Snapshot directory does not exist"
        }), 404

    return send_from_directory(
        folder,
        filename
    )


# ============================================================
# Category-specific APIs
# ============================================================

@app.route("/api/snapshots/people")
def people_snapshots():
    """
    Return only People snapshots.
    """

    return jsonify({
        "category": "people",
        "snapshots": get_images(
            "people",
            SNAPSHOT_DIRS["people"]
        )
    })


@app.route("/api/snapshots/vehicles")
def vehicle_snapshots():
    """
    Return only Vehicle snapshots.
    """

    return jsonify({
        "category": "vehicles",
        "snapshots": get_images(
            "vehicles",
            SNAPSHOT_DIRS["vehicles"]
        )
    })


@app.route("/api/snapshots/others")
def others_snapshots():
    """
    Return:

        Animals
        Electronics
        Other

    under the single Others category.
    """

    images = []

    for category in (
        "animals",
        "electronics",
        "other"
    ):

        images.extend(
            get_images(
                category,
                SNAPSHOT_DIRS[category]
            )
        )

    images.sort(
        key=lambda item: item["timestamp"],
        reverse=True
    )

    if MAX_IMAGES_PER_CATEGORY is not None:
        images = images[
            :MAX_IMAGES_PER_CATEGORY
        ]

    return jsonify({
        "category": "others",
        "snapshots": images
    })


# ============================================================
# Error Handlers
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({
        "error": "Endpoint not found"
    }), 404


@app.errorhandler(500)
def internal_error(error):

    return jsonify({
        "error": "Internal server error"
    }), 500


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Jetson AI Snapshot Server")
    print("=" * 60)

    print(f"Base snapshot directory:")
    print(f"  {BASE_DIR}")

    print()
    print("Snapshot directories:")

    for category, folder in SNAPSHOT_DIRS.items():
        exists = os.path.isdir(folder)

        print(
            f"  {category:12} -> "
            f"{folder} "
            f"[{'OK' if exists else 'NOT FOUND'}]"
        )

    print()
    print(f"Server:")
    print(f"  http://{HOST}:{PORT}")

    print()
    print("API:")
    print(f"  http://<JETSON-IP>:{PORT}/api/health")
    print(f"  http://<JETSON-IP>:{PORT}/api/snapshots")
    print(f"  http://<JETSON-IP>:{PORT}/api/snapshots/count")

    print()
    print("=" * 60)

    app.run(
        host=HOST,
        port=PORT,
        debug=False,
        threaded=True
    )

