import streamlit as st
import numpy as np
import utils
import cv2
st.set_page_config(page_title='OpenCV', page_icon='📷', layout='wide', initial_sidebar_state='collapsed')
code = """ 
# Converting CV images to and from buffers
np.asarray(bytearray(buffer.read()), dtype="uint8")
cv2.imencode('.jpg', scaled_image)[1].tobytes()
cv2.imdecode(image, cv2.IMREAD_COLOR)

# Resize Image
cv2.resize(grey_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

# Convert Color
cv2.cvtColor(opencv_image, cv2.COLOR_BGR2HSV)

# Canny Edge Detection

# Erosion and Dilation
"""
def app():
    _, mid, _ = st.columns([0.7,1.15,0.5])
    with mid:
        st.title('First principles!')

    st.subheader('Begin your exploration')    
    uploaded_file = st.file_uploader("Upload an image:", type="jpg")

    curious = st.sidebar.checkbox('Show code')

    if uploaded_file is not None:
        # Convert the file to an opencv image.
        opencv_image = utils.get_opencv_img_from_buffer(uploaded_file)
        # Now do something with the image! For example, let's display it:

        with st.expander('Raw Image'):
            scale_factor = st.slider('Select raw resize factor', min_value = 1.0, max_value = 5.0, step=0.5)
            scaled_raw = cv2.resize(opencv_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            st.write(f'Image Shape: {scaled_raw.shape}')
            st.image(scaled_raw, channels="BGR", caption='Input Image', use_column_width=False)
            btn = st.download_button(
                label="Download image",
                data=cv2.imencode('.jpg', scaled_raw)[1].tobytes(),
                file_name="image.jpg",
                mime="image/jpg"
           )

        with st.expander('HSV Image'):
            hsv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2HSV)
            scale_factor = st.slider('Select hsv resize factor', min_value = 0.01, max_value = 5.0, step=0.01)
            scaled_hsv = cv2.resize(hsv_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            st.write(f'Image Shape: {scaled_hsv.shape}')
            st.image(scaled_hsv, caption='HSV Image', use_column_width=False)
            btn = st.download_button(
                label="Download image",
                data=cv2.imencode('.jpg', scaled_hsv)[1].tobytes(),
                file_name="image.jpg",
                mime="image/jpg"
           )

        with st.expander('Grey Scale Image'):
            grey_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            scale_factor = st.slider('Select grey scale resize factor', min_value = 1.0, max_value = 5.0, step=0.5)
            scaled_grey = cv2.resize(grey_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            st.write(f'Image Shape: {scaled_grey.shape}')
            st.image(scaled_grey, caption='Grey Scale Image', use_column_width=False)
            btn = st.download_button(
                label="Download image",
                data=cv2.imencode('.jpg', scaled_grey)[1].tobytes(),
                file_name="image.jpg",
                mime="image/jpg"
           )

        with st.expander('Image Edge Detection'):
            st.write('**Detector Configuration**')
            min_threshold = st.slider('Select minimum threshold to apply', min_value = 1.0, max_value = 500.0, step=0.5)
            max_threshold = st.slider('Select maximum threshold to apply', min_value = 1.0, max_value = 500.0, step=0.5, value=min_threshold)

            st.write('**Errosion and Dilation Configuration**')
            iterations = st.slider('Select number of iterations for errosion and dilation', min_value = 1, max_value = 10, step=1)
            kernel_dim = st.slider('Select the size of Kernel to user for errosion and dilation', min_value = 1, max_value=10, step=1)
            
            edge_img = cv2.Canny(scaled_grey, min_threshold, max_threshold)
            kernel = np.ones((kernel_dim, kernel_dim),np.uint8)

            st.write(f'Image Shape: {edge_img.shape}')
            st.write(f'Kernal used for errosion and dilation: {kernel}')

            erroded_img = cv2.erode(edge_img, kernel, iterations=iterations)
            dilated_img = cv2.dilate(edge_img, kernel, iterations=iterations)
    
            left, mid, right = st.columns(3)

            with left:
                st.image(erroded_img, caption='Erroded edges', use_column_width=True)

                btn = st.download_button(
                    label="Download image",
                    data=cv2.imencode('.jpg', erroded_img)[1].tobytes(),
                    file_name="erroded_image.jpg",
                    mime="image/jpg"
            )
            with mid:
                st.image(edge_img, caption='Image edges', use_column_width=True)

                btn = st.download_button(
                    label="Download image",
                    data=cv2.imencode('.jpg', edge_img)[1].tobytes(),
                    file_name="edge_image.jpg",
                    mime="image/jpg"
            )
            with right:
                st.image(dilated_img, caption='Dilated edges', use_column_width=True)

                btn = st.download_button(
                    label="Download image",
                    data=cv2.imencode('.jpg', dilated_img)[1].tobytes(),
                    file_name="dilated_image.jpg",
                    mime="image/jpg"
            )
            
    if curious:
        st.code(code, language='python')
    
app()