from ultralytics import YOLO
import os
import sys

import streamlit as st
from PIL import Image


model = YOLO("./runs/detect/train2/weights/best.pt")


def get_dish_name(class_id):
    with open("./dataset/labels/train/classes.txt", 'r') as f:
        lines = f.readlines()
        if 0 < class_id < len(lines):
            return lines[class_id].strip("\n")
        else:
            print("Error in get_dish_name()")

nutrition_data = {
    "roti":        {"cal": 120, "protein": 3.6, "fat": 3.7, "carbs": 18},
    "dal":         {"cal": 198, "protein": 11.2, "fat": 6.4, "carbs": 22.4},
    "rice":        {"cal": 130, "protein": 2.7, "fat": 0.3, "carbs": 28},
    "pakora":      {"cal": 315, "protein": 9,   "fat": 22,  "carbs": 22},
    "samosa":      {"cal": 252, "protein": 3,   "fat": 17,  "carbs": 19},
    "maggi":       {"cal": 380, "protein": 9.5, "fat": 14,  "carbs": 55},
    "pizza":       {"cal": 266, "protein": 11,  "fat": 10,  "carbs": 33},
    "paneer":      {"cal": 265, "protein": 18,  "fat": 21,  "carbs": 2},
    "salad":       {"cal": 80,  "protein": 2,   "fat": 1,   "carbs": 16},
    "aloo sabzi":  {"cal": 160, "protein": 3,   "fat": 7,   "carbs": 21},
    "pasta":       {"cal": 131, "protein": 5,   "fat": 1.1, "carbs": 25},
    "idli":        {"cal": 60,  "protein": 2,   "fat": 0.4, "carbs": 13},
    "biryani":     {"cal": 300, "protein": 6,   "fat": 10,  "carbs": 40},
    "dosa":        {"cal": 133, "protein": 3,   "fat": 4,   "carbs": 19}
}



def get_nutri(file_name):
    results = model.predict(source=file_name, save=False, show=False, conf=0.5)
    dish_names = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            dish_names.append(get_dish_name(class_id))
    
    return dish_names


st.set_page_config(
    page_title="NutriSnap - AI Food Analyzer",
    page_icon="🍽️", 
    # layout="wide"
)

st.title("🍽️ NutriSnap")
st.write("Upload your food image to analyze nutrients")

uploaded_file = st.file_uploader("Upload a plate image", type=["jpg", "jpeg", "png"])
UPLOAD_FOLDER = "uploaded_images"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# st.title("Upload & Save Image")


def get_feedback(cal, protein, fat, carbs):
    feedback = []

    # Calorie feedback
    if cal < 300:
        feedback.append("🍽️ Very low calories. Add more carbs (rice/roti).")
    elif cal > 800:
        feedback.append("⚠️ High calorie meal. Consider reducing oily items.")

    # Protein feedback
    if protein < 10:
        feedback.append("💪 Protein is low! Add dal/paneer/rajma/chicken.")
    elif protein > 25:
        feedback.append("🔥 Great protein intake.")

    # Fat feedback
    if fat > 20:
        feedback.append("🥲 Too much fat. Avoid fried items like samosa/pakora.")

    # Carbs feedback
    if carbs < 20:
        feedback.append("⚡ Carbs are low. Add rice/roti for energy.")
    elif carbs > 60:
        feedback.append("🍞 Carbs are high. Balance it with protein.")

    if len(feedback) == 0:
        feedback.append("✅ Perfect balanced meal!")

    return feedback


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    image.save(save_path)
    st.success("✅ Image uploaded successfully!")
    # print(get_nutri(save_path))
    total_cal = 0
    total_protein = 0
    total_fat = 0
    total_carbs = 0
    dishes = list(set(get_nutri(save_path)))
    if dishes:
        st.header("🥗 Nutrients in your plate")
        for dish in dishes:
            nutri = nutrition_data[dish]
            st.markdown(f"""
                ### 🍛 {dish.upper()}
                🔸 **Calories:** {nutri['cal']} kcal  
                🔸 **Protein:** {nutri['protein']} g  
                🔸 **Fat:** {nutri['fat']} g  
                🔸 **Carbs:** {nutri['carbs']} g  
                """)
            total_cal += nutri['cal']
            total_protein += nutri['protein']
            total_fat += nutri['fat']
            total_carbs += nutri['carbs']

        feedback_list = get_feedback(total_cal, total_protein, total_fat, total_carbs)
        st.subheader("🧠 Nutrition Feedback")
        for fb in feedback_list:
            st.write("- ", fb)
    else:
        st.markdown(f"""
            ### ⚠️ No food detected in the image
            &nbsp;Try uploading a clearer image with visible food items.
            """)
      
