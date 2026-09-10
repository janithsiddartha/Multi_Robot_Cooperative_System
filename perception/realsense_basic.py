import pyrealsense2 as rs
import numpy as np
import cv2


def main():
    # Create RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable RGB and Depth
    config.enable_stream(
        rs.stream.depth,
        1280,
        720,
        rs.format.z16,
        30
    )

    config.enable_stream(
        rs.stream.color,
        1280,
        720,
        rs.format.bgr8,
        30
    )

    # Start camera
    profile = pipeline.start(config)

    print("\n=== RealSense D456 ===")

    device = profile.get_device()
    print("Device:", device.get_info(rs.camera_info.name))
    print("Serial:", device.get_info(rs.camera_info.serial_number))
    print("Firmware:",
          device.get_info(rs.camera_info.firmware_version))

    # Depth sensor information
    depth_sensor = device.first_depth_sensor()

    depth_scale = depth_sensor.get_depth_scale()

    print("Depth scale:", depth_scale, "meters/unit")

    # Get stream profiles
    depth_profile = profile.get_stream(
        rs.stream.depth
    ).as_video_stream_profile()

    color_profile = profile.get_stream(
        rs.stream.color
    ).as_video_stream_profile()

    depth_intrinsics = depth_profile.get_intrinsics()
    color_intrinsics = color_profile.get_intrinsics()

    print("\nDepth intrinsics:")
    print("  Width :", depth_intrinsics.width)
    print("  Height:", depth_intrinsics.height)
    print("  fx    :", depth_intrinsics.fx)
    print("  fy    :", depth_intrinsics.fy)
    print("  cx    :", depth_intrinsics.ppx)
    print("  cy    :", depth_intrinsics.ppy)

    print("\nColor intrinsics:")
    print("  Width :", color_intrinsics.width)
    print("  Height:", color_intrinsics.height)
    print("  fx    :", color_intrinsics.fx)
    print("  fy    :", color_intrinsics.fy)
    print("  cx    :", color_intrinsics.ppx)
    print("  cy    :", color_intrinsics.ppy)

    print("\nStarting stream...")
    print("Press Q to quit.\n")

    try:
        while True:
            # Wait for synchronized frames
            frames = pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert to NumPy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Convert depth to displayable image
            depth_display = cv2.convertScaleAbs(
                depth_image,
                alpha=0.03
            )

            depth_colormap = cv2.applyColorMap(
                depth_display,
                cv2.COLORMAP_JET
            )

            # Show images
            cv2.imshow("D456 RGB", color_image)
            cv2.imshow("D456 Depth", depth_colormap)

            # Quit with Q
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
