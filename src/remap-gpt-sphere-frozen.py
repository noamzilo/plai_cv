import base64
import os
import webbrowser
from pathlib import Path
from threading import Timer

import cv2
import dash
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

# --- Configuration ---
INPUT_FOLDER = "images"
OUTPUT_FOLDER = "processed_images3"
IMAGE_FILENAME = (
    "IMG20250425185749.jpg"  # "IMG20250425185710.jpg"  # image to overlay and render
)
MARKER_SIZE_M = 0.2  # physical marker side length in meters
FLIP_VERTICAL = True  # flip capture vertically if needed
ARUCO_DICT = cv2.aruco.DICT_5X5_1000
PORT = 8050  # Dash port
FOV_DEG = 84.42  # horizontal FOV for focal estimate
TEXTURE_WIDTH = 640  # Target width in pixels for the texture plane image

# Ensure output folder exists
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)


# --- ArUco Detector Setup ---
def make_detector():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT), params
    )


detector = make_detector()


# --- Marker Detection ---
def detect_markers(img):
    corners, ids, _ = detector.detectMarkers(img)
    if ids is None:
        return [], np.empty((0, 1), dtype=int)
    return corners, ids


# --- Build 3D Scene ---
def build_scene(img, corners, ids):
    # --- Check if input image is valid ---
    if img is None or img.size == 0:
        raise ValueError("Invalid input image provided to build_scene")

    # --- Use original dimensions for intrinsics calculation ---
    h, w = img.shape[:2]
    f = w / (2 * np.tan(np.deg2rad(FOV_DEG / 2)))
    print(f)
    # NOTE: Ideally, K and dist should come from camera calibration
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=float)
    dist = np.zeros((5,), dtype=float)  # Assuming zero distortion for now

    # --- Undistort the original image using K and dist ---
    # This corrects for lens distortion before we use the image as a texture.
    # Using K as the new camera matrix provides a standard view.
    print("Undistorting input image...")
    try:
        img_undistorted = cv2.undistort(img, K, dist, None, K)
        print("Image undistorted successfully.")
    except cv2.error as e:
        print(f"Error during undistortion: {e}")
        # Fallback to original image if undistortion fails
        img_undistorted = img.copy()

    # --- Resize the *undistorted* image for texture ---
    h_orig, w_orig = img_undistorted.shape[:2]  # Use undistorted dimensions
    aspect_ratio = h_orig / w_orig
    target_w = TEXTURE_WIDTH
    target_h = int(target_w * aspect_ratio)
    try:
        # Resize the UNDISTORTED image
        img_resized = cv2.resize(
            img_undistorted, (target_w, target_h), interpolation=cv2.INTER_AREA
        )
    except cv2.error as e:
        print(f"Error resizing undistorted image: {e}")
        raise  # Re-raise the exception
    print(f"Resized undistorted texture image to: {target_w}x{target_h}")

    # Object points in marker frame (meters)
    s = MARKER_SIZE_M
    objp = np.array(
        [
            [-s / 2, -s / 2, 0],
            [s / 2, -s / 2, 0],
            [s / 2, s / 2, 0],
            [-s / 2, s / 2, 0],
        ],
        dtype=float,
    )

    ids_flat = ids.flatten().tolist()
    # Reorder to put 20 at the beginning of the list
    planar_idxs = [i for i, m in enumerate(ids_flat) if m == 20] + [
        i for i, m in enumerate(ids_flat) if m != 20 and m != 40
    ]
    if not planar_idxs:
        raise RuntimeError("Need at least one planar marker ID != 40 for ground")
    ref_idx = planar_idxs[0]
    print("origin:", ids_flat[ref_idx])

    # Solve PnP using original corner coordinates, K, and dist
    _, rvec_ref, tvec_ref = cv2.solvePnP(objp, corners[ref_idx], K, dist)
    R_ref, _ = cv2.Rodrigues(rvec_ref)
    Rcw = R_ref.T
    tcw = (-Rcw @ tvec_ref).flatten()

    # Compute focal distance from ArUco 20
    if 20 in ids_flat:
        idx_20 = ids_flat.index(20)
        marker_corners = corners[idx_20][0]  # shape (4,2)
        # Compute average side length in pixels
        side_lengths = [
            np.linalg.norm(marker_corners[i] - marker_corners[(i + 1) % 4])
            for i in range(4)
        ]
        marker_size_px = np.mean(side_lengths)
        # Distance from camera to marker (in meters)
        marker_distance_m = np.linalg.norm(tvec_ref)
        # Focal length in pixels
        f_pixels_aruco = (marker_size_px * marker_distance_m) / MARKER_SIZE_M
        print(f"Focal length estimated from ArUco 20: {f_pixels_aruco:.2f} pixels")
        # If you want focal length in meters and know the physical image width:
        # f_meters_aruco = f_pixels_aruco * (w_meters / w)
        # print(f"Focal length in meters: {f_meters_aruco:.4f} m")
        # You can now use marker_distance_m as the focal distance for your 3D geometry if you wish:
        plane_distance = 0.4  # marker_distance_m
        print(plane_distance)
    else:
        print("ArUco 20 not found, using default plane_distance.")
        plane_distance = 0.5  # fallback

    # --- NaN Check for Camera Pose ---
    if np.any(np.isnan(tcw)) or np.any(np.isnan(Rcw)):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"ERROR: NaN detected in camera pose! tcw={tcw}, Rcw=\n{Rcw}")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Handle error, maybe raise exception or return empty figure
        raise ValueError("NaN detected in camera pose")

    # Camera orientation vectors & NaN Check
    forward = (Rcw @ np.array([0, 0, 1], float)).flatten()
    cam_y_axis = (Rcw @ np.array([0, 1, 0], float)).flatten()
    cam_x_axis = (Rcw @ np.array([1, 0, 0], float)).flatten()
    up_vec = -cam_y_axis
    if (
        np.any(np.isnan(forward))
        or np.any(np.isnan(cam_x_axis))
        or np.any(np.isnan(cam_y_axis))
    ):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(
            f"ERROR: NaN detected in camera vectors! fwd={forward}, x={cam_x_axis}, y={cam_y_axis}"
        )
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        raise ValueError("NaN detected in camera orientation vectors")

    forward_magnitude = np.linalg.norm(forward)
    up_vec_magnitude = np.linalg.norm(up_vec)

    print("Forward vector magnitude:", forward_magnitude)
    print("Up vector magnitude:", up_vec_magnitude)

    traces = []
    marker_ray_trace_indices = {}  # Store {marker_id: trace_index}
    camera_trace_index = -1  # Initialize

    # Plot all markers and their rays
    print(ids_flat)
    for idx, m in enumerate(ids_flat):
        adjust_aruco = True
        if adjust_aruco:
            ok, rvec, tvec = cv2.solvePnP(objp, corners[idx], K, dist)
            if not ok:
                continue
            R_m, _ = cv2.Rodrigues(rvec)
            Rw = Rcw @ R_m
            original_tw = (Rcw @ (tvec - tvec_ref)).flatten()  # Original position

            # --- Adjust position along camera ray ---
            dir_vector = original_tw - tcw  # Direction from camera to marker
            desired_z = -0.92 if m == 40 else 0.0  # Set desired height

            if dir_vector[2] != 0:
                t = (desired_z - tcw[2]) / dir_vector[2]
                new_tw = tcw + dir_vector * t
                new_tw[2] = desired_z  # Ensure exact height
            else:
                new_tw = original_tw.copy()
                new_tw[2] = desired_z

            tw = new_tw  # Use adjusted position
        else:
            ok, rvec, tvec = cv2.solvePnP(objp, corners[idx], K, dist)
            if not ok:
                continue
            R_m, _ = cv2.Rodrigues(rvec)
            # Calculate marker pose in the World frame (relative to marker 20)
            Rw = Rcw @ R_m
            tw = (Rcw @ (tvec - tvec_ref)).flatten()
        height = tw[2]  # Height of the marker from the ground
        print(f"Marker ID {m} height from ground: {height:.2f} meters")  # Print height

        # Visualization adjustments (Z=0 for ground, Z=1 for marker 40)
        adjust_40 = False
        if adjust_40:
            if m == 40:
                tw[2] = -0.92
                # Optional reorientation for marker 40
                vec = tcw - tw
                vec_norm = np.linalg.norm(vec)
                if vec_norm > 1e-6:
                    vec /= vec_norm
                right = np.cross(up_vec, vec)
                right_norm = np.linalg.norm(right)
                if right_norm > 1e-6:
                    right /= right_norm
                actual_up = np.cross(vec, right)
                Rw = np.column_stack((right, actual_up, vec))
            else:  # For marker 20 and others on the ground
                tw[2] = 0.0

        # Calculate world coordinates of corners
        cw = (Rw @ objp.T).T + tw
        xs = np.append(cw[:, 0], cw[0, 0])
        ys = np.append(cw[:, 1], cw[0, 1])
        zs = np.append(cw[:, 2], cw[0, 2])
        # Add the trace for the marker itself
        traces.append(
            go.Scatter3d(x=xs, y=ys, z=zs, mode="lines+markers", name=f"Marker {m}")
        )

        # --- Group Projection Rays for THIS marker ---
        print(f"Grouping rays for marker {m}.")  # Debug print
        ray_xs, ray_ys, ray_zs = [], [], []
        valid_corners_count = 0
        # Determine ray color based on marker ID
        if m == 20:
            ray_color = "cyan"
        elif m == 40:
            ray_color = "magenta"
        else:
            ray_color = "gray"  # Default color for other markers

        for i in range(4):  # Loop through the 4 corners
            corner_coord = cw[i, :]
            # Add NaN check for corner coordinates before drawing ray
            if np.any(np.isnan(corner_coord)) or np.any(np.isinf(corner_coord)):
                print(
                    f"Warning: Invalid corner coordinate for marker {m}, corner {i+1}. Skipping ray segment."
                )
                continue

            # Add line segment (camera -> corner) coordinates
            # Add None between segments to break the line in Plotly
            ray_xs.extend([tcw[0], corner_coord[0], None])
            ray_ys.extend([tcw[1], corner_coord[1], None])
            ray_zs.extend([tcw[2], corner_coord[2], None])
            valid_corners_count += 1

        # Add the single combined ray trace for this marker if any valid corners were found
        if valid_corners_count > 0:
            ray_trace_name = f"Rays Marker {m}"
            traces.append(
                go.Scatter3d(
                    x=ray_xs,
                    y=ray_ys,
                    z=ray_zs,
                    mode="lines",
                    line=dict(
                        color=ray_color, width=2
                    ),  # Increased width for clickability
                    name=ray_trace_name,
                    showlegend=True,  # Show this group in the legend
                    visible=False,  # Start visible
                )
            )
            # Store the index of this trace using the marker ID as the key
            marker_ray_trace_indices[m] = len(traces) - 1
        else:
            print(f"Skipping ray trace for marker {m} due to invalid corner data.")
        # --- End Group Projection Rays ---

    # --- Spherical Patch (Image as a patch on the sphere) ---
    MESH_RESOLUTION = 200
    print(f"Using MESH_RESOLUTION: {MESH_RESOLUTION}")

    # Calculate plane dimensions using ORIGINAL image dimensions (w, h) and K (f)
    plane_width_m = 2 * plane_distance * np.tan(np.deg2rad(FOV_DEG / 2))
    plane_height_m = plane_width_m * (h / w) if w > 0 else 0

    # Compute angular extents for the patch
    half_h_fov = np.arctan((plane_height_m / 2) / plane_distance)
    half_w_fov = np.arctan((plane_width_m / 2) / plane_distance)

    n_theta = MESH_RESOLUTION + 1
    n_phi = MESH_RESOLUTION + 1

    # theta: vertical angle from center (0) to top/bottom
    theta = np.linspace(-half_h_fov, half_h_fov, n_theta)
    # phi: horizontal angle from center (0) to left/right
    phi = np.linspace(-half_w_fov, half_w_fov, n_phi)
    phi_grid, theta_grid = np.meshgrid(phi, theta)

    # Spherical coordinates (z is forward)
    x_s = plane_distance * np.sin(phi_grid) * np.cos(theta_grid)
    y_s = plane_distance * np.sin(theta_grid)
    z_s = plane_distance * np.cos(phi_grid) * np.cos(theta_grid)

    # Stack and rotate to world coordinates
    sphere_points = np.stack([x_s, y_s, z_s], axis=-1).reshape(-1, 3)
    R_cam = np.column_stack(
        [
            cam_x_axis / np.linalg.norm(cam_x_axis),
            cam_y_axis / np.linalg.norm(cam_y_axis),
            forward / np.linalg.norm(forward),
        ]
    )
    sphere_points_world = (sphere_points @ R_cam.T) + tcw
    xw = sphere_points_world[:, 0].reshape(x_s.shape)
    yw = sphere_points_world[:, 1].reshape(y_s.shape)
    zw = sphere_points_world[:, 2].reshape(z_s.shape)

    # Prepare the texture image
    img_texture = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    h_tex, w_tex = img_texture.shape[:2]

    # Map spherical patch to image texture
    # Normalize phi/theta to [0, 1] for texture coordinates
    u = (phi_grid - phi_grid.min()) / (phi_grid.max() - phi_grid.min())
    v = (theta_grid - theta_grid.min()) / (theta_grid.max() - theta_grid.min())

    vertex_colors_list = []
    for vi in range(n_theta):
        for ui in range(n_phi):
            px = min(max(0, int(u[vi, ui] * (w_tex - 1))), w_tex - 1)
            py = min(max(0, int(v[vi, ui] * (h_tex - 1))), h_tex - 1)
            color_rgb = img_texture[py, px]
            hex_color = "#%02x%02x%02x" % (color_rgb[0], color_rgb[1], color_rgb[2])
            vertex_colors_list.append(hex_color)

    # Define faces (triangles) for the mesh grid
    mesh_i, mesh_j, mesh_k = [], [], []
    for r in range(n_theta - 1):
        for c in range(n_phi - 1):
            idx_tl = r * n_phi + c
            idx_tr = idx_tl + 1
            idx_bl = (r + 1) * n_phi + c
            idx_br = idx_bl + 1
            # Triangle 1 (TL, TR, BL)
            mesh_i.append(idx_tl)
            mesh_j.append(idx_tr)
            mesh_k.append(idx_bl)
            # Triangle 2 (TR, BR, BL)
            mesh_i.append(idx_tr)
            mesh_j.append(idx_br)
            mesh_k.append(idx_bl)

    # Add the spherical patch as a mesh
    traces.append(
        go.Mesh3d(
            x=xw.flatten(),
            y=yw.flatten(),
            z=zw.flatten(),
            i=mesh_i,
            j=mesh_j,
            k=mesh_k,
            vertexcolor=vertex_colors_list,
            opacity=1.0,
            flatshading=True,
            name="Image Spherical Patch",
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=0.0),
            showlegend=True,
        )
    )

    # Plot camera as a marker (diamond shape) at the camera position
    camera_trace_index = len(traces)  # Store index BEFORE adding camera trace
    traces.append(
        go.Scatter3d(
            x=[tcw[0]],
            y=[tcw[1]],
            z=[tcw[2]],
            mode="markers+text",
            marker=dict(size=12, colorscale="Viridis", symbol="diamond"),
            text=["Camera"],
            textposition="bottom center",
            name="Camera",  # Assign a specific name
        )
    )

    # Plot forward direction as an arrow
    arrow_end = tcw + forward  # Arrow endpoint is in the forward direction
    traces.append(
        go.Cone(
            x=[tcw[0]],
            y=[tcw[1]],
            z=[tcw[2]],
            u=[forward[0]],
            v=[forward[1]],
            w=[forward[2]],
            sizemode="absolute",
            sizeref=0.1,  # Control size of the cone
            colorscale="Viridis",
            showscale=False,
            name="Forward Arrow",
            anchor="tail",
        )
    )

    # Calculate the forward direction endpoint
    forward_end = tcw + forward * 10  # Extend the line in the forward direction

    # Ensure the z-coordinate is set to 0 for the ground plane
    # Calculate the intersection with the ground plane (z=0)
    if forward[2] != 0:  # Avoid division by zero
        t = -tcw[2] / forward[2]  # Calculate the scalar to reach z=0
        intersection_point = tcw + forward * t  # Calculate the intersection point
    else:
        intersection_point = (
            forward_end  # If forward direction is horizontal, use forward_end
        )

    # Append the trace for the forward direction
    traces.append(
        go.Scatter3d(
            x=[tcw[0], intersection_point[0]],
            y=[tcw[1], intersection_point[1]],
            z=[tcw[2], 0],  # Ground plane (z=0)
            mode="lines",
            line=dict(color="red", width=4),
            name="Forward Direction",
        )
    )

    # Ground plane
    x_grid = np.linspace(-5, 5, 10)  # 10 points from -5 to 5
    y_grid = np.linspace(-7, 3, 10)  # 10 points from -7 to 3
    xx, yy = np.meshgrid(x_grid, y_grid)  # Create the meshgrid
    traces.append(
        go.Surface(
            x=xx, y=yy, z=np.zeros_like(xx), opacity=0.3, showscale=False, name="Ground"
        )
    )

    # --- Add lines on the ground ---
    line_width = 2  # Adjustable width for the lines
    # Line at y = 0 from x = -5 to x = 5
    traces.append(
        go.Scatter3d(
            x=[-5, 5],
            y=[-0.25, 0],
            z=[0, 0],
            mode="lines",
            line=dict(color="black", width=line_width),
            name="Line at y=0",
        )
    )
    # Line at x = 0 from y = -7 to y = 3
    traces.append(
        go.Scatter3d(
            x=[-0.1, -0.2],
            y=[-7, 3],
            z=[0, 0],
            mode="lines",
            line=dict(color="black", width=line_width),
            name="Line at x=0",
        )
    )

    # --- Add square around the ground ---
    square_width = 10  # Adjustable width for the square
    half_width = square_width / 2
    square_x = [-half_width, half_width, half_width, -half_width, -half_width]
    square_y = [-7, -7, 3, 3, -7]
    square_z = [0, 0, 0, 0, 0]  # All points at z=0

    traces.append(
        go.Scatter3d(
            x=square_x,
            y=square_y,
            z=square_z,
            mode="lines",
            line=dict(color="black", width=line_width),
            name="Square around Ground",
        )
    )

    # Setup Plotly camera
    eye_pos = tcw - forward * 0.01
    center_pos = tcw + forward
    cam_cfg = dict(
        eye=dict(x=eye_pos[0], y=eye_pos[1], z=eye_pos[2]),
        center=dict(x=center_pos[0], y=center_pos[1], z=center_pos[2]),
        up=dict(x=up_vec[0], y=up_vec[1], z=up_vec[2]),
        projection=dict(type="perspective"),
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="3D ArUco Scene with Projected Image",
        scene_camera=dict(
            eye=dict(x=2, y=2, z=2),  # Change these values to get the desired view
            up=dict(x=0, y=1, z=0),  # +Y is up
            center=dict(x=0, y=0, z=0),
        ),
        scene=dict(
            xaxis_title="X (right of ArUco 20)",
            yaxis_title="Y (up of ArUco 20)",
            zaxis_title="Z (out of ArUco 20)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    # Return the figure, camera config, position, and the new index maps
    return fig, cam_cfg, tcw, marker_ray_trace_indices, camera_trace_index


# --- Main ---
def main():
    # Annotate

    # Load and build
    img = cv2.imread(os.path.join(INPUT_FOLDER, IMAGE_FILENAME))
    if img is None:
        raise FileNotFoundError(f"Image '{IMAGE_FILENAME}' not found")
    img = cv2.flip(img, 1) if FLIP_VERTICAL else img.copy()

    corners, ids = detect_markers(img)
    if ids is None or not corners:  # Check for valid detection results
        raise RuntimeError(f"No markers detected in '{IMAGE_FILENAME}'")
    if ids.size:
        cv2.aruco.drawDetectedMarkers(img, corners, ids)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, IMAGE_FILENAME), img)

    # Unpack the new return values from build_scene
    fig3d, camera_view, tcw, marker_ray_indices, camera_trace_idx = build_scene(
        img, corners, ids
    )

    # Encode overlay
    _, buf = cv2.imencode(".jpg", img)
    img_src = f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"

    initial_eye = camera_view["eye"]
    initial_user = f"User View Position: {initial_eye['x']:.2f}, {initial_eye['y']:.2f}, {initial_eye['z']:.2f}"

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.Div(
                [
                    dcc.Graph(id="scene", figure=fig3d, style={"height": "100vh"}),
                    html.Img(
                        id="overlay",
                        src=img_src,
                        style={
                            "position": "absolute",
                            "top": 0,
                            "left": 0,
                            "width": "100%",
                            "height": "100vh",
                            "opacity": 0,
                            "pointer-events": "none",
                        },
                    ),
                ],
                style={"position": "relative"},
            ),
            html.Div(
                [
                    html.Label("Blend Opacity"),
                    dcc.Slider(id="opacity-slider", min=0, max=1, step=0.01, value=0.5),
                ],
                style={
                    "position": "fixed",
                    "bottom": "20px",
                    "left": "20px",
                    "width": "300px",
                },
            ),
            html.Div(
                [
                    html.H3("Scene Information"),
                    html.P(initial_user, id="user-view-position"),
                    html.P(
                        "Cursor 3D Position: No point selected", id="cursor-position"
                    ),
                ],
                style={
                    "position": "fixed",
                    "bottom": "100px",
                    "left": "20px",
                    "background": "white",
                    "padding": "10px",
                    "border": "1px solid black",
                },
            ),
        ]
    )

    Timer(1, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}"))

    # Store the original figure and indices for the callback context
    # Use a dictionary to hold potentially changing state
    callback_context = {
        "fig_initial": fig3d,
        "marker_ray_indices": marker_ray_indices,  # Dict {marker_id: trace_index}
        "camera_trace_idx": camera_trace_idx,
        "hidden_ray_traces": set(),  # Keep track of hidden ray trace indices (start empty)
    }

    @app.callback(
        [
            Output("overlay", "style"),
            Output("scene", "figure"),
            Output("cursor-position", "children"),
        ],
        Input("scene", "clickData"),
        State("opacity-slider", "value"),
    )
    def update_view(click, opacity):
        # --- Get context data ---
        # Use a fresh copy of the initial figure to apply changes
        fig_current = go.Figure(callback_context["fig_initial"])
        marker_rays_map = callback_context[
            "marker_ray_indices"
        ]  # Dict {marker_id: trace_index}
        cam_idx = callback_context["camera_trace_idx"]
        # Make a mutable copy of the hidden set for this callback run
        hidden_rays_current = callback_context["hidden_ray_traces"].copy()

        # --- Defaults ---
        style = {  # Default overlay style (hidden)
            "position": "absolute",
            "top": 0,
            "left": 0,
            "width": "100%",
            "height": "100vh",
            "opacity": 0,
            "pointer-events": "none",
        }
        cursor_txt = "Cursor 3D Position: No point selected"
        reset_camera_view = False
        figure_changed = False  # Flag to check if figure needs returning

        # --- Process Click ---
        if click and click.get("points"):
            p = click["points"][0]
            clicked_trace_index = p.get("curveNumber")

            # Validate index and get trace name
            if clicked_trace_index is not None and clicked_trace_index < len(
                fig_current.data
            ):
                clicked_trace_name = fig_current.data[clicked_trace_index].name
            else:
                clicked_trace_name = None
                clicked_trace_index = None

            # Update cursor position text
            if all(k in p for k in ("x", "y", "z")):
                cursor_txt = (
                    f"Cursor 3D Position: {p['x']:.2f}, {p['y']:.2f}, {p['z']:.2f}"
                )

            # --- Interaction Logic ---
            if clicked_trace_index is not None:  # Proceed only if a trace was clicked
                figure_changed = True  # Assume a change might occur

                # 1. Clicked on Camera? -> Reset all visibility, show overlay, reset view
                if clicked_trace_index == cam_idx or clicked_trace_name == "Camera":
                    print("Camera clicked, showing overlay and resetting view.")
                    style["opacity"] = opacity
                    reset_camera_view = True
                    hidden_rays_current.clear()  # Make all ray groups visible

                # 2. Clicked on a Ray Group? -> Toggle its visibility
                elif clicked_trace_name and clicked_trace_name.startswith(
                    "Rays Marker"
                ):
                    print(
                        f"Clicked on Ray Group: {clicked_trace_name} (Index: {clicked_trace_index}). Toggling visibility."
                    )
                    if clicked_trace_index in hidden_rays_current:
                        hidden_rays_current.remove(clicked_trace_index)  # Show it
                        print(f"  Showing trace {clicked_trace_index}.")
                    else:
                        hidden_rays_current.add(clicked_trace_index)  # Hide it
                        print(f"  Hiding trace {clicked_trace_index}.")

                # 3. Clicked on a Marker Square? -> Toggle associated Ray Group visibility
                elif clicked_trace_name and clicked_trace_name.startswith("Marker "):
                    try:
                        marker_id_str = clicked_trace_name.split(" ")[-1]
                        marker_id = int(marker_id_str)
                        print(
                            f"Clicked on Marker: {clicked_trace_name} (ID: {marker_id}). Toggling associated rays."
                        )

                        if marker_id in marker_rays_map:
                            corresponding_ray_trace_index = marker_rays_map[marker_id]
                            if corresponding_ray_trace_index in hidden_rays_current:
                                hidden_rays_current.remove(
                                    corresponding_ray_trace_index
                                )  # Show rays
                                print(
                                    f"  Showing associated rays (Trace Index: {corresponding_ray_trace_index})."
                                )
                            else:
                                hidden_rays_current.add(
                                    corresponding_ray_trace_index
                                )  # Hide rays
                                print(
                                    f"  Hiding associated rays (Trace Index: {corresponding_ray_trace_index})."
                                )
                        else:
                            print(f"  No associated rays found for marker {marker_id}.")
                            figure_changed = False  # No change needed
                    except (ValueError, IndexError, KeyError) as e:
                        print(
                            f"Error processing marker click for '{clicked_trace_name}': {e}"
                        )
                        figure_changed = False  # No change on error

                # 4. Clicked on something else? -> No visibility action
                else:
                    print(
                        f"Clicked on non-interactive trace: '{clicked_trace_name}' (Index: {clicked_trace_index}). No action."
                    )
                    figure_changed = False  # No change needed

            else:  # Clicked on background
                print("Click detected, but not on a specific trace.")
                figure_changed = False  # No change needed

        # --- Apply Visibility State ---
        # Update trace visibility based on the final hidden_rays_current set
        print(f"Applying visibility. Hidden set: {hidden_rays_current}")
        visibility_applied = False
        for trace_idx, trace in enumerate(fig_current.data):
            is_ray_trace = trace.name and trace.name.startswith("Rays Marker")
            if is_ray_trace:
                should_be_visible = trace_idx not in hidden_rays_current
                # Only update if the state actually changes
                if trace.visible != should_be_visible:
                    trace.visible = should_be_visible
                    visibility_applied = True  # A visual change occurred

        # Determine if the figure actually needs to be returned
        figure_changed = figure_changed or visibility_applied or reset_camera_view

        # --- Update Camera ---
        if reset_camera_view:
            print("Resetting camera view.")
            fig_current.update_layout(scene_camera=camera_view)

        # --- Update Persistent State ---
        # Store the final hidden set back into the context if it changed
        if hidden_rays_current != callback_context["hidden_ray_traces"]:
            callback_context["hidden_ray_traces"] = hidden_rays_current

        # --- Return Results ---
        # Use dash.no_update for the figure if no visual changes occurred
        final_figure = fig_current if figure_changed else dash.no_update
        return style, final_figure, cursor_txt

    @app.callback(
        Output("user-view-position", "children"),
        Input("scene", "relayoutData"),
    )
    def update_user(relayout_data):
        if relayout_data and "scene.camera" in relayout_data:
            eye = relayout_data["scene.camera"]["eye"]
            return f"User View Position: {eye['x']:.2f}, {eye['y']:.2f}, {eye['z']:.2f}"
        return dash.no_update

    app.run(debug=True, host="127.0.0.1", port=PORT)


if __name__ == "__main__":
    main()
