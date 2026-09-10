import pyrealsense2 as rs
import numpy as np
import cv2
import time


# ============================================================
# RealSense D456
# Spatial + Temporal Robust 3D Measurement
# ============================================================


# ------------------------------------------------------------
# Camera configuration
# ------------------------------------------------------------

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.depth,
    1280, 720,
    rs.format.z16,
    30
)

config.enable_stream(
    rs.stream.color,
    1280, 720,
    rs.format.bgr8,
    30
)

profile = pipeline.start(config)

#device = profile.get_device()

#depth_sensor = device.first_depth_sensor()

#if depth_sensor.supports(
 #   rs.option.visual_preset
#:

#    depth_sensor.set_option(
#        rs.option.visual_preset,
#        5
#    )

   # print("\nVisual preset: Medium Density (5)")


# ------------------------------------------------------------
# Camera information
# ------------------------------------------------------------

color_profile = profile.get_stream(
    rs.stream.color
).as_video_stream_profile()

color_intrinsics = color_profile.get_intrinsics()

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()


print("\n========================================")
print("RealSense D456")
print("Spatial + Temporal 3D Measurement")
print("========================================")

print(f"Depth scale: {depth_scale} m/unit")

print("\nColor Intrinsics:")
print(f"fx = {color_intrinsics.fx}")
print(f"fy = {color_intrinsics.fy}")
print(f"cx = {color_intrinsics.ppx}")
print(f"cy = {color_intrinsics.ppy}")

print("\nControls:")
print("  LEFT CLICK -> measure 3D point")
print("  Q          -> quit")
print("========================================\n")


# ------------------------------------------------------------
# Align depth to color
# ------------------------------------------------------------

align = rs.align(rs.stream.color)


# ------------------------------------------------------------
# Robust measurement parameters
# ------------------------------------------------------------

NEIGHBORHOOD_RADIUS = 2       # 5x5 neighborhood

TEMPORAL_FRAMES = 15          # number of frames collected

MIN_VALID_PIXELS = 5          # minimum valid pixels in 5x5

MAX_SPATIAL_SPREAD = 0.05     # 5 cm

MAX_TEMPORAL_SPREAD = 0.10    # 10 cm


# ------------------------------------------------------------
# Mouse
# ------------------------------------------------------------

selected_pixel = None


def mouse_callback(event, x, y, flags, param):

    global selected_pixel

    if event == cv2.EVENT_LBUTTONDOWN:

        selected_pixel = (x, y)


# ------------------------------------------------------------
# Window
# ------------------------------------------------------------

cv2.namedWindow("D456 RGB")
cv2.setMouseCallback(
    "D456 RGB",
    mouse_callback
)


# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------

try:

    while True:

        # ----------------------------------------------------
        # Get frame
        # ----------------------------------------------------

        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(
            color_frame.get_data()
        )

        depth_image = np.asanyarray(
            depth_frame.get_data()
        )


        # ----------------------------------------------------
        # Measurement requested
        # ----------------------------------------------------

        if selected_pixel is not None:

            x, y = selected_pixel

            print("\n========================================")
            print("MEASUREMENT STARTED")
            print(f"Selected pixel: ({x}, {y})")
            print(f"Collecting {TEMPORAL_FRAMES} frames...")
            print("========================================")


            temporal_depths = []


            # =================================================
            # Collect multiple frames
            # =================================================

            for frame_number in range(
                TEMPORAL_FRAMES
            ):

                # ---------------------------------------------
                # Get a fresh frame
                # ---------------------------------------------

                frames = pipeline.wait_for_frames()

                aligned_frames = align.process(frames)

                depth_frame = (
                    aligned_frames.get_depth_frame()
                )

                if not depth_frame:
                    continue

                depth_image = np.asanyarray(
                    depth_frame.get_data()
                )


                # ---------------------------------------------
                # 5x5 neighborhood
                # ---------------------------------------------

                r = NEIGHBORHOOD_RADIUS

                x_min = max(0, x - r)
                x_max = min(
                    depth_image.shape[1],
                    x + r + 1
                )

                y_min = max(0, y - r)
                y_max = min(
                    depth_image.shape[0],
                    y + r + 1
                )

                neighborhood = depth_image[
                    y_min:y_max,
                    x_min:x_max
                ]


                # ---------------------------------------------
                # Depth → metres
                # ---------------------------------------------

                depth_values = (
                    neighborhood.astype(np.float32)
                    * depth_scale
                )


                # ---------------------------------------------
                # Remove invalid depth
                # ---------------------------------------------

                valid_depth = depth_values[
                    depth_values > 0
                ]


                # ---------------------------------------------
                # Check spatial validity
                # ---------------------------------------------

                if len(valid_depth) < MIN_VALID_PIXELS:

                    print(
                        f"Frame {frame_number + 1:02d}: "
                        "INVALID - insufficient depth"
                    )

                    continue


                spatial_min = np.min(valid_depth)
                spatial_max = np.max(valid_depth)

                spatial_spread = (
                    spatial_max - spatial_min
                )


                if (
                    spatial_spread
                    > MAX_SPATIAL_SPREAD
                ):

                    print(
                        f"Frame {frame_number + 1:02d}: "
                        f"REJECTED - spatial spread "
                        f"{spatial_spread:.3f} m"
                    )

                    continue


                # ---------------------------------------------
                # Spatial median
                # ---------------------------------------------

                frame_depth = float(
                    np.median(valid_depth)
                )

                temporal_depths.append(
                    frame_depth
                )


                print(
                    f"Frame {frame_number + 1:02d}: "
                    f"depth = {frame_depth:.4f} m"
                )


            # =================================================
            # Temporal validation
            # =================================================

            print("\n----------------------------------------")

            valid_count = len(temporal_depths)

            print(
                f"Valid temporal samples: "
                f"{valid_count}/{TEMPORAL_FRAMES}"
            )


            if valid_count < 5:

                print(
                    "REJECTED: Not enough valid "
                    "temporal measurements."
                )

            else:

                temporal_depths = np.array(
                    temporal_depths,
                    dtype=np.float32
                )


                temporal_min = np.min(
                    temporal_depths
                )

                temporal_max = np.max(
                    temporal_depths
                )

                temporal_spread = (
                    temporal_max
                    - temporal_min
                )

                temporal_median = float(
                    np.median(temporal_depths)
                )


                print(
                    f"Temporal depth range: "
                    f"{temporal_min:.4f} - "
                    f"{temporal_max:.4f} m"
                )

                print(
                    f"Temporal spread: "
                    f"{temporal_spread:.4f} m"
                )

                print(
                    f"Temporal median: "
                    f"{temporal_median:.4f} m"
                )


                # =================================================
                # Final stability check
                # =================================================

                if (
                    temporal_spread
                    > MAX_TEMPORAL_SPREAD
                ):

                    print(
                        "REJECTED: Temporal depth unstable."
                    )

                    print(
                        f"Allowed spread: "
                        f"{MAX_TEMPORAL_SPREAD:.3f} m"
                    )


                    cv2.putText(
                        color_image,
                        "UNSTABLE DEPTH",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                else:

                    # =================================================
                    # Final 3D deprojection
                    # =================================================

                    point_3d = (
                        rs.rs2_deproject_pixel_to_point(
                            color_intrinsics,
                            [x, y],
                            temporal_median
                        )
                    )

                    Xc, Yc, Zc = point_3d


                    print("\n========================================")
                    print("FINAL VALID 3D MEASUREMENT")
                    print("========================================")

                    print(
                        f"Pixel: ({x}, {y})"
                    )

                    print(
                        f"Temporal samples: "
                        f"{valid_count}"
                    )

                    print(
                        f"Temporal median depth: "
                        f"{Zc:.4f} m"
                    )

                    print(
                        f"Camera XYZ:"
                    )

                    print(
                        f"Xc = {Xc:.4f} m"
                    )

                    print(
                        f"Yc = {Yc:.4f} m"
                    )

                    print(
                        f"Zc = {Zc:.4f} m"
                    )

                    print("========================================")


                    # ---------------------------------------------
                    # Draw selected point
                    # ---------------------------------------------

                    cv2.circle(
                        color_image,
                        (x, y),
                        7,
                        (0, 255, 0),
                        -1
                    )


                    # ---------------------------------------------
                    # Display result
                    # ---------------------------------------------

                    text1 = (
                        f"XYZ = "
                        f"({Xc:.3f}, "
                        f"{Yc:.3f}, "
                        f"{Zc:.3f}) m"
                    )

                    text2 = (
                        f"Depth = "
                        f"{Zc:.3f} m"
                    )

                    cv2.putText(
                        color_image,
                        text1,
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        color_image,
                        text2,
                        (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )


            print("----------------------------------------\n")


            # ------------------------------------------------
            # Reset click
            # ------------------------------------------------

            selected_pixel = None


        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        cv2.imshow(
            "D456 RGB",
            color_image
        )


        # ----------------------------------------------------
        # Keyboard
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break


finally:

    pipeline.stop()
    cv2.destroyAllWindows()

    print("\nCamera stopped.")