import os
import cv2
import time
import uuid
import shutil
import pickle
import yaml
import subprocess
import numpy as np
import pandas as pd
import sqlite3
from datetime import date
from collections import defaultdict
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
# Import your custom modules
from sql_table import face_data_pipeline
from yolo_split_pipeline import process_single_category_folder
from ultralytics import YOLO
import cvzone
import av
# ----------------- PAGE CONFIGURATION & GLOBAL STYLE -----------------
st.set_page_config(page_title="Live Member Face Recognition", layout="wide")
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .stApp { background-color: #0d1117; }
    h1, h2, h3, h4, h5 { color: #58a6ff; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        color: #c9d1d9;
        border: none;
        padding: 10px 20px;
        border-radius: 5px 5px 0 0;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"]:hover { background-color: #21262d; color: #58a6ff; }
    .stTabs [aria-selected="true"] { background-color: #238636; color: white; }
    .stForm {
        padding: 1rem;
        background-color: #161b22;
        border-radius: 8px;
        border: 1px solid #30363d;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 6px;
        padding: 0.5em 1em;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        transition: 0.3s ease-in-out;
    }
    .stTextInput>div>div>input,
    .stNumberInput input,
    .stSelectbox div div,
    .stDateInput input {
        background-color: #161b22;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 4px;
    }
    .stAlert {
        background-color: #161b22 !important;
        border-left: 6px solid #238636 !important;
        color: #c9d1d9 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- PIPELINE & DYNAMIC CLASS NAMES -----------------
pipeline = face_data_pipeline("capstone")
table_name = "small_members"

# Load dynamic class names from new_data.yaml (if it exists)
data_yaml_path = "new_data.yaml"
if os.path.exists(data_yaml_path):
    with open(data_yaml_path, "r") as f:
        data_yaml = yaml.safe_load(f)
    dynamic_classNames = data_yaml.get("names", [])
else:
    dynamic_classNames = []  # fallback if file not found

# ----------------- GLOBAL FIELD NAMES FOR DISPLAY -----------------
FIELD_NAMES = [
    "ID", "Name", "Age", "Date of Birth", "Address", "Loyalty",
    "Member Since", "Gender", "Email", "Phone Number", "Membership Type",
    "Status", "Occupation", "Interests", "Marital Status"
]
field_mapping = {
    "ID": "id",
    "Name": "name",
    "Age": "age",
    "Date of Birth": "date_of_birth",
    "Address": "address",
    "Loyalty": "loyalty",
    "Member Since": "member_since",
    "Gender": "gender",
    "Email": "email",
    "Phone Number": "phone_number",
    "Membership Type": "membership_type",
    "Status": "status",
    "Occupation": "occupation",
    "Interests": "interests",
    "Marital Status": "marital_status"
}

# Global constants for persisting detection data in predictions
COUNTS_FILE = "final_counts.pkl"
FRAMES_FILE = "final_frames.pkl"

# ----------------- SIDEBAR SETTINGS (Sliders & Selection) -----------------
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.01)
font_threshold = st.sidebar.slider("Font Scale", 0.0, 5.0, 0.7, 0.1)
thickness_threshold = st.sidebar.slider("Text Thickness", 0, 10, 1, 1)
line_threshold = st.sidebar.slider("Line Height (px)", 0, 100, 30, 1)
selected_fields = st.sidebar.multiselect("Select Fields to Display", options=FIELD_NAMES, default=FIELD_NAMES)

# ----------------- TAB DEFINITION -----------------
tab_names = [
    "➕ Add Member",
    "📸 Add Images",
    "🛠️ Build Model",
    "🧠 Predictions Webcam",
    "🎥 Predictions Upload Video"
]
tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_names)

# ================= TAB 1: ADD MEMBER & MEMBER MANAGEMENT =================
with tab1:
    st.title("📋 Add a New Member")
    try:
        last_id = pipeline.max_id(table_name)
    except Exception:
        last_id = 0
    new_id = last_id + 1

    with st.form("member_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.text_input("ID (Auto-assigned)", value=str(new_id), disabled=True)
            name = st.text_input("Name")
            age = st.number_input("Age", min_value=0)
            dob = st.date_input("Date of Birth", value=date(2000, 1, 1))
        with col2:
            address = st.text_input("Address")
            loyalty = st.number_input("Loyalty Points", min_value=0)
            member_since = st.date_input("Member Since", value=date.today())
            gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])
        with col3:
            email = st.text_input("Email")
            phone = st.text_input("Phone Number")
            membership_type = st.selectbox("Membership Type", ["", "Silver", "Gold", "Platinum"])
            status = st.selectbox("Status", ["", "Active", "Inactive"])

        occupation = st.text_input("Occupation")
        interests = st.text_input("Interests (comma-separated)")
        marital_status = st.selectbox("Marital Status", ["", "Single", "Married", "Divorced", "Widowed"])
        submitted = st.form_submit_button("Submit")
        if submitted:
            if not all([new_id, name.strip(), age, address.strip(), loyalty,
                        gender.strip(), email.strip(), phone.strip(),
                        membership_type.strip(), status.strip(),
                        occupation.strip(), interests.strip(), marital_status.strip()]):
                st.error("⚠️ All fields must be filled out completely.")
            elif pipeline.name_exists(table_name, name.strip()):
                st.error(f"❌ The name '{name}' already exists.")
            else:
                member_data = {
                    "id": new_id,
                    "name": name.strip(),
                    "age": age,
                    "date_of_birth": dob.strftime("%Y-%m-%d"),
                    "address": address.strip(),
                    "loyalty": loyalty,
                    "member_since": member_since.strftime("%Y-%m-%d"),
                    "gender": gender,
                    "email": email.strip(),
                    "phone_number": phone.strip(),
                    "membership_type": membership_type,
                    "status": status,
                    "occupation": occupation.strip(),
                    "interests": interests.strip(),
                    "marital_status": marital_status
                }
                try:
                    pipeline.add_member(table_name, member_data)
                    st.success(f"✅ Member '{name}' added successfully!")
                    st.session_state.new_member = name.strip()
                    if os.path.exists("new_data.yaml"):
                        with open("new_data.yaml", "r") as f:
                            yaml_content = f.read()
                        st.write("**Current new_data.yaml content:**")
                        st.code(yaml_content, language="yaml")
                    else:
                        st.info("new_data.yaml does not exist yet.")
                except Exception as e:
                    st.error(f"❌ Failed to insert: {e}")

    st.markdown("## Delete Member")
    try:
        members_df = pipeline.view_all(table_name)
        if not members_df.empty:
            members_options = {f"{row['id']} - {row['name']}": row['id'] for index, row in members_df.iterrows()}
        else:
            members_options = {}
    except Exception as e:
        st.error("Could not retrieve members from database.")
        members_options = {}

    if members_options:
        member_to_delete = st.selectbox("Select a member to delete", list(members_options.keys()))
        if st.button("Delete Member"):
            member_id = members_options[member_to_delete]
            try:
                pipeline.delete_member(table_name, member_id)
                st.success(f"Member with ID {member_id} deleted successfully.")
                yaml_data_path = "new_data.yaml"
                if os.path.exists(yaml_data_path):
                    with open(yaml_data_path, "r") as f:
                        data_yaml = yaml.safe_load(f)
                    names_list = data_yaml.get("names", [])
                    deleted_name = member_to_delete.split(" - ", 1)[1]
                    if deleted_name in names_list:
                        names_list.remove(deleted_name)
                        data_yaml["names"] = names_list
                        data_yaml["nc"] = len(names_list)
                        with open(yaml_data_path, "w") as f:
                            yaml.dump(data_yaml, f, default_flow_style=False)
                        st.success("Updated new_data.yaml after deletion.")
                        with open(yaml_data_path, "r") as f:
                            yaml_content = f.read()
                        st.write("**Current new_data.yaml content:**")
                        st.code(yaml_content, language="yaml")
                else:
                    st.info("new_data.yaml does not exist yet.")
            except Exception as ex:
                st.error(f"Failed to delete member: {ex}")
    else:
        st.info("No members available for deletion.")

# ================= TAB 2: ADD IMAGES (WEB CAM CAPTURE) & RESET FACE DATASET =================
with tab2:
    st.title("📸 Capture Full Webcam Images")
    from twilio.rest import Client
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    if account_sid and auth_token:
        client = Client(account_sid, auth_token)
        try:
            token = client.tokens.create()
            ice_servers = token.ice_servers
        except Exception as e:
            st.error(f"Error creating TURN token: {e}")
            ice_servers = [{"urls": ["stun:stun1.l.google.com:19302"]}]
    else:
        st.warning("Twilio credentials not set. Using free STUN server.")
        ice_servers = [{"urls": ["stun:stun1.l.google.com:19302"]}]
    
    FACE_DIR = "face_dataset"
    # Use a capture resolution similar to Tab 4 (1280x720)
    IMG_WIDTH = 1280
    IMG_HEIGHT = 720
    TOTAL_IMAGES = 150

    def create_person_folder(name):
        folder_name = f"person_{name.replace(' ', '_')}"
        path = os.path.join(FACE_DIR, folder_name)
        os.makedirs(path, exist_ok=True)
        return path

    class FaceCaptureProcessor(VideoProcessorBase):
        def __init__(self):
            self.name = None
            self.path = None
            self.image_count = 0
            self.start_time = None  # For a 5-second delay before capturing

        def set_name(self, name):
            self.name = name
            self.path = create_person_folder(name)

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            if self.start_time is None:
                self.start_time = time.time()
            elapsed = time.time() - self.start_time
            if elapsed < 5:
                # Wait 5 seconds before starting capture; show countdown.
                seconds_left = int(5 - elapsed)
                cv2.putText(img, f"Adjust camera... Starting in {seconds_left}s", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            # Do not resize the image; use it directly. had issues with this
            if self.image_count < TOTAL_IMAGES:
                filename = f"{uuid.uuid4().hex}.jpg"
                filepath = os.path.join(self.path, filename)
                cv2.imwrite(filepath, img)
                self.image_count += 1
                cv2.putText(img, f"{self.image_count}/{TOTAL_IMAGES}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            else:
                cv2.putText(img, "Done collecting images!", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

    if "new_member" in st.session_state:
        selected_name = st.session_state.new_member
        st.info(f"Capturing {TOTAL_IMAGES} images for **{selected_name}**.")
        processor = FaceCaptureProcessor()
        processor.set_name(selected_name)
        webrtc_streamer(
            key=f"capture_{selected_name}",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: processor,
            rtc_configuration={"iceServers": ice_servers},
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280},
                    "height": {"ideal": 720},
                    "frameRate": {"ideal": 30},
                    "aspectRatio": 1.777778,
                    "facingMode": "user"
                },
                "audio": False
            },
            async_processing=True
        )
    else:
        st.warning("No new member has been added in this session. Please add a new member in Tab 1!")
    
    st.markdown("### Reset Entire Face Dataset")
    if st.button("Reset Face Dataset"):
        try:
            if os.path.exists("face_dataset"):
                shutil.rmtree("face_dataset")
                st.success("Entire face_dataset folder has been deleted.")
            else:
                st.info("face_dataset folder does not exist.")
        except Exception as e:
            st.error(f"Failed to delete face_dataset folder: {e}")

# ================= TAB 3: BUILD YOLO DATASET & TRAIN MODEL =================
with tab3:
    st.title("🛠️ Build YOLO Dataset and Train Model")
    if "new_member" in st.session_state:
        selected_name_model = st.session_state.new_member
        st.info(f"Preparing YOLO dataset for **{selected_name_model}**...")
        class_id = pipeline.max_id(table_name) - 1
        folder_path = os.path.join("face_dataset", f"person_{selected_name_model.replace(' ', '_')}")
        
        if st.button("🚀 Build YOLO Dataset"):
            with st.spinner("Processing and building dataset..."):
                process_single_category_folder(
                    folder_path,
                    "my_data_split",
                    {"train": 0.7, "valid": 0.2, "test": 0.1},
                    class_id=class_id
                )
                for subset in ["train", "valid", "test"]:
                    images_path = os.path.join("my_data_split", subset, "images")
                    if not os.path.exists(images_path):
                        os.makedirs(images_path, exist_ok=True)
                try:
                    project_root = os.getcwd()
                    st.info(f"DEBUG: Project root: {project_root}")
                    abs_train = os.path.join(project_root, "my_data_split", "train", "images").replace("\\", "/")
                    abs_val   = os.path.join(project_root, "my_data_split", "valid", "images").replace("\\", "/")
                    abs_test  = os.path.join(project_root, "my_data_split", "test", "images").replace("\\", "/")
                    
                    yaml_data_path = "new_data.yaml"
                    if os.path.exists(yaml_data_path):
                        with open(yaml_data_path, "r") as f:
                            data_yaml = yaml.safe_load(f)
                        names_list = data_yaml.get("names", [])
                        if selected_name_model not in names_list:
                            names_list.append(selected_name_model)
                    else:
                        all_members = pipeline.view_all(table_name).sort_values("id")
                        names_list = all_members["name"].tolist()
                    
                    class_count = len(names_list)
                    yaml_data = {
                        "train": abs_train,
                        "val": abs_val,
                        "test": abs_test,
                        "nc": class_count,
                        "names": names_list
                    }
                    with open(yaml_data_path, "w") as f:
                        yaml.dump(yaml_data, f, default_flow_style=False)
                    st.success("✅ YOLO dataset and new_data.yaml successfully updated.")
                    with open(yaml_data_path, "r") as f:
                        yaml_content = f.read()
                    st.write("**Updated new_data.yaml content:**")
                    st.code(yaml_content, language="yaml")
                except Exception as e:
                    st.error(f"❌ Failed to update new_data.yaml: {e}")
                
                st.info("📦 Now starting YOLOv8 model training with freezing (freeze=15)...")
                if os.path.exists("new_person_detect.pt"):
                    model_to_use = "new_person_detect.pt"
                else:
                    model_to_use = "best.pt"
                st.info(f"DEBUG: Using model for training: {model_to_use}")
                
                try:
                    result = subprocess.run([
                        "yolo", "task=detect", "mode=train", f"model={model_to_use}", "data=new_data.yaml",
                        "epochs=10", "imgsz=320", "freeze=15", 
                        "project=runs/detect", "name=train", "exist_ok=True"
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        st.success("✅ YOLO model training complete.")
                        st.code(result.stdout)
                        source_path = os.path.join(os.getcwd(), "runs", "detect", "train", "weights", "best.pt")
                        st.info("DEBUG: New training weights should be located at: " + source_path)
                        
                        if os.path.exists(source_path):
                            new_weight_filename = f"person_{selected_name_model.replace(' ', '_')}.pt"
                            dest_path = os.path.join(os.getcwd(), new_weight_filename)
                            shutil.copy(source_path, dest_path)
                            st.success(f"✅ New weights for {selected_name_model} stored as '{new_weight_filename}'.")
                            st.info(f"DEBUG: New weights copied from {source_path} to {dest_path}")
                            
                            fixed_dest = os.path.join(os.getcwd(), "new_person_detect.pt")
                            shutil.copy(source_path, fixed_dest)
                            st.info("DEBUG: Cumulative weights updated in 'new_person_detect.pt'.")
                            
                            try:
                                os.remove(source_path)
                                st.info("DEBUG: Deleted training weights file at " + source_path)
                            except Exception as e:
                                st.error(f"Failed to delete training weights file at {source_path}: {e}")
                        else:
                            st.error("❌ New training weights file not found in the expected location.")
                    else:
                        st.error("❌ YOLO model training failed.")
                        st.code(result.stderr)
                        st.error("DEBUG: YOLO training error details printed above.")
                except Exception as e:
                    st.error(f"❌ Exception during training: {e}")
                    st.error("DEBUG: Exception details printed above.")
                
                if os.path.exists("face_dataset"):
                    try:
                        shutil.rmtree("face_dataset")
                        st.success("Deleted folder 'face_dataset'.")
                    except Exception as e:
                        st.error(f"Failed to delete folder 'face_dataset': {e}")
                if os.path.exists("runs"):
                    if os.path.exists("old_run"):
                        shutil.rmtree("old_run")
                    shutil.move("runs", "old_run")
    else:
        st.warning("No new member available for building the dataset. Please add a new member in Tab 1.")

# ================= TAB 4: PREDICTIONS WEBCAM =================
def getProfile(member_id):
    conn = sqlite3.connect('capstone.db')
    conn.row_factory = sqlite3.Row
    cmd = "SELECT * FROM small_members WHERE id=?"
    cursor = conn.execute(cmd, (member_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

weights_path = "new_person_detect.pt" if os.path.exists("new_person_detect.pt") else "best.pt"
st.write("DEBUG: Using weight file for predictions: " + weights_path)
model = YOLO(weights_path)
classNames = dynamic_classNames

with tab4:
    st.title("Predictions Webcam")
    from twilio.rest import Client
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    if account_sid and auth_token:
        client = Client(account_sid, auth_token)
        try:
            token = client.tokens.create()
            ice_servers = token.ice_servers
        except Exception as e:
            st.error(f"Error creating TURN token: {e}")
            ice_servers = [{"urls": ["stun:stun1.l.google.com:19302"]}]
    else:
        st.warning("Twilio credentials not set. Using free STUN server.")
        ice_servers = [{"urls": ["stun:stun1.l.google.com:19302"]}]

    class YOLOVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.prev_frame_time = time.time()
            self.detection_counts = defaultdict(int)
            self.detected_frames = {}
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            new_frame_time = time.time()
            detections_this_frame = []
            results = model(img, conf=confidence_threshold, stream=True)
            highest_conf_info = None
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < confidence_threshold:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    profile = getProfile(cls + 1)
                    if profile is not None:
                        detected_name = profile.get(field_mapping["Name"], "Unknown")
                        detected_status = profile.get(field_mapping["Status"], "Inactive")
                    else:
                        detected_name = classNames[cls] if 0 <= cls < len(classNames) else "Unknown"
                        detected_status = "Inactive"
                    self.detection_counts[detected_name] += 1
                    conf_percent = int(conf * 100)
                    background_color = (0, 255, 0) if detected_status.lower() == "active" else (0, 0, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), background_color, thickness_threshold)
                    label_text = f'{detected_name} - {detected_status} {conf_percent}%'
                    cvzone.putTextRect(img, label_text, (max(0, x1), max(35, y1)),
                                       scale=font_threshold * 3,
                                       thickness=thickness_threshold, colorT=(0, 0, 0),
                                       colorR=background_color)
                    if highest_conf_info is None or conf > highest_conf_info['confidence']:
                        highest_conf_info = {'name': detected_name, 'status': detected_status, 'confidence': conf}
                    if profile is not None:
                        startY = y1 + (y2 - y1) + 20
                        for field in selected_fields:
                            key = field_mapping.get(field)
                            if key and key in profile:
                                text = f"{field}: {profile[key]}"
                                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_threshold, thickness_threshold)
                                cv2.rectangle(img, (x1, startY), (x1 + text_w, startY + text_h + 10), (0, 0, 0), cv2.FILLED)
                                cv2.putText(img, text, (x1, startY + text_h), cv2.FONT_HERSHEY_SIMPLEX, font_threshold, (0, 255, 0), thickness_threshold)
                                startY += line_threshold
                    detections_this_frame.append((detected_name, x1, y1, x2, y2, profile))
            if highest_conf_info is not None:
                global_text = f"{highest_conf_info['name']} - {highest_conf_info['status']}"
                global_color = (0, 255, 0) if highest_conf_info['status'].lower() == "active" else (0, 0, 255)
                cv2.putText(img, global_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, font_threshold*2.5, global_color, 3)
            for (det_name, x1, y1, x2, y2, profile) in detections_this_frame:
                self.detected_frames[det_name] = (img.copy(), profile)
            with open(COUNTS_FILE, "wb") as f:
                pickle.dump(dict(self.detection_counts), f)
            with open(FRAMES_FILE, "wb") as f:
                pickle.dump(self.detected_frames, f)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    ctx = webrtc_streamer(
         key="webcam",
         video_processor_factory=lambda: YOLOVideoProcessor(),
         rtc_configuration={"iceServers": ice_servers},
         media_stream_constraints={
             "video": {
                 "width": {"ideal": 1280},
                 "height": {"ideal": 720},
                 "frameRate": {"ideal": 30},
                 "aspectRatio": 1.777778,
                 "facingMode": "user"
             },
             "audio": False
         },
         async_processing=True
    )
    if ctx is not None and ctx.state is not None and not ctx.state.playing:
         st.write("Webcam stream stopped.")
         if os.path.exists(COUNTS_FILE) and os.path.exists(FRAMES_FILE):
             with open(COUNTS_FILE, "rb") as f:
                 final_counts = pickle.load(f)
             with open(FRAMES_FILE, "rb") as f:
                 final_frames = pickle.load(f)
         else:
             final_counts = {}
             final_frames = {}
         if st.button("Show Results"):
             if final_counts:
                 df = pd.DataFrame(list(final_counts.items()), columns=["Name", "Detections"])
                 df["Confidence %"] = round(confidence_threshold * 100, 2)
                 df["Status"] = df["Name"].apply(lambda x: "Active" if x.lower() in [n.lower() for n in classNames] else "Inactive")
                 st.table(df)
                 csv = df.to_csv(index=False).encode("utf-8")
                 st.download_button("Download Results", data=csv, file_name="detection_results.csv", mime="text/csv")
             else:
                 st.write("No detections to display.")
             if final_counts:
                 max_name = max(final_counts, key=final_counts.get)
                 max_count = final_counts[max_name]
                 st.write(f"**Highest Detection: {max_name} ({max_count} times)**")
                 if max_name in final_frames:
                     full_img, profile = final_frames[max_name]
                     desired_width = 800
                     scale = desired_width / full_img.shape[1]
                     resized_img = cv2.resize(full_img, (desired_width, int(full_img.shape[0] * scale)))
                     overlay_height = 150
                     overlay = np.zeros((overlay_height, desired_width, 3), dtype=np.uint8)
                     if profile is not None:
                         lines = []
                         for field in selected_fields:
                             key = field_mapping.get(field)
                             if key and key in profile:
                                 lines.append(f"{field}: {profile[key]}")
                         y0 = 30
                         dy = 25
                         for idx, line in enumerate(lines):
                             cv2.putText(overlay, line, (10, y0 + idx * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                     final_img = np.vstack((resized_img, overlay))
                     st.markdown("### Final Annotated Image (Highest Detection)")
                     st.image(final_img, channels="BGR", use_column_width=True)
                     cv2.imwrite(f"{max_name}_final_full.jpg", final_img)
                 else:
                     st.write("No captured image for the highest detection.")

# ================= TAB 5: PREDICTIONS UPLOAD VIDEO =================
with tab5:
    st.title("Predictions Upload Video")
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"], key="video_uploader")
    rotate_mode = st.selectbox("Rotate Video", ["None", "90° CW", "180°", "90° CCW", "Flip Horizontal"], key="rotate_video")
    adjust = st.slider("Adjust Scale", 0.1, 2.0, 1.0, 0.1, key="adjust_scale")
    out_width = st.number_input("Output Width", min_value=100, value=640, key="out_width")
    out_height = st.number_input("Output Height", min_value=100, value=480, key="out_height")
    process_button = st.button("Process Video", key="process_video")
    
    def process_video_file(input_bytes, rotate_mode, adjust, output_size,
                             conf_threshold, font_threshold, thickness_threshold, line_threshold):
        input_path = "temp_input_video.mp4"
        output_path = "temp_output_video.mp4"
        with open(input_path, "wb") as f:
            f.write(input_bytes.read())
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            st.error("Error opening video file.")
            return None
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        st.info(f"Original video size: {orig_width} x {orig_height}, FPS: {fps}")
        out_w, out_h = output_size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
        detection_counts_vid = defaultdict(int)
        detected_frames_vid = {}
        preview_placeholder = st.empty()
        prev_frame_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if rotate_mode == "90° CW":
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotate_mode == "180°":
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotate_mode == "90° CCW":
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rotate_mode == "Flip Horizontal":
                frame = cv2.flip(frame, 1)
            new_frame_time = time.time()
            highest_conf_value = 0
            highest_conf_info = None
            detections_this_frame = []
            results = model(frame, conf=conf_threshold, stream=True)
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < conf_threshold:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    profile = getProfile(cls + 1)
                    if profile is not None:
                        detected_name = profile.get(field_mapping["Name"], "Unknown")
                        detected_status = profile.get(field_mapping["Status"], "Inactive")
                    else:
                        detected_name = classNames[cls] if 0 <= cls < len(classNames) else "Unknown"
                        detected_status = "Inactive"
                    detection_counts_vid[detected_name] += 1
                    conf_percent = int(conf * 100)
                    label_text = f"{detected_name} - {detected_status} {conf_percent}%"
                    box_color = (0, 255, 0) if detected_status.lower() == "active" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness_threshold)
                    cvzone.putTextRect(frame, label_text, (max(0, x1), max(35, y1)),
                                       scale=font_threshold * 2,
                                       thickness=thickness_threshold,
                                       colorR=box_color)
                    if conf > highest_conf_value:
                        highest_conf_value = conf
                        highest_conf_info = {'name': detected_name, 'status': detected_status}
                    if profile is not None:
                        startY = y1 + (y2 - y1) + 20
                        for field in selected_fields:
                            key = field_mapping.get(field)
                            if key and key in profile:
                                t = f"{field}: {profile[key]}"
                                (text_w, text_h), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, font_threshold, thickness_threshold)
                                cv2.rectangle(frame, (x1, startY), (x1 + text_w, startY + text_h + 10),
                                              (0, 0, 0), cv2.FILLED)
                                cv2.putText(frame, t, (x1, startY + text_h),
                                            cv2.FONT_HERSHEY_SIMPLEX, font_threshold, (0, 255, 0), thickness_threshold)
                                startY += line_threshold
                    detections_this_frame.append((detected_name, x1, y1, x2, y2))
            if highest_conf_info is not None:
                global_text = f"{highest_conf_info['name']} - {highest_conf_info['status']}"
                global_color = (0, 255, 0) if highest_conf_info['status'].lower() == "active" else (0, 0, 255)
                cv2.putText(frame, global_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, font_threshold * 2.5, global_color, 3)
            for (det_name, x1, y1, x2, y2) in detections_this_frame:
                face_crop = frame[y1:y2, x1:x2].copy()
                detected_frames_vid[detected_name] = (frame.copy(), face_crop)
            out_frame = cv2.resize(frame, (out_w, out_h))
            out.write(out_frame)
            fps_val = 1 / (new_frame_time - prev_frame_time) if prev_frame_time else 0
            prev_frame_time = new_frame_time
            fps_text = f"FPS: {fps_val:.2f}"
            (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            pos = (frame.shape[1] - tw - 10, th + 10)
            cv2.putText(frame, fps_text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
            preview_frame = cv2.resize(frame, (0, 0), fx=adjust, fy=adjust) if adjust != 1.0 else frame
            preview_placeholder.image(preview_frame, channels="BGR")
        cap.release()
        out.release()
        preview_placeholder.empty()
        st.success("Video processing complete!")
        with open(output_path, "rb") as vid_file:
            st.download_button("Download Processed Video", data=vid_file, file_name=output_path, mime="video/mp4")
        if detection_counts_vid:
            df_vid = pd.DataFrame(list(detection_counts_vid.items()), columns=["Name", "Detections"])
            df_vid["Confidence %"] = round(conf_threshold * 100, 2)
            df_vid["Status"] = df_vid["Name"].apply(lambda x: "Active" if x.lower() in [n.lower() for n in classNames] else "Inactive")
            st.table(df_vid)
        return detection_counts_vid, detected_frames_vid

    if uploaded_video is not None and process_button:
        detection_counts_vid, detected_frames_vid = process_video_file(
            uploaded_video,
            rotate_mode,
            adjust,
            (int(out_width), int(out_height)),
            confidence_threshold,
            font_threshold,
            thickness_threshold,
            line_threshold
        )
