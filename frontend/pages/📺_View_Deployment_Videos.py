import boto3
import streamlit as st

from sftk.common import S3_BUCKET


# --- Helper to generate a presigned URL ---
def get_presigned_url(key: str, expires_in: int = 3600) -> str:
    s3 = boto3.client("s3")
    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=expires_in,
    )


# --- Simple password gate ---
def check_password():
    """Returns True if the user entered the correct password."""

    if "password_correct" not in st.session_state:
        # First run, ask for the password.
        st.session_state.password_correct = False

    if not st.session_state.password_correct:
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if password == st.secrets["APP_PASSWORD"]:
                st.session_state.password_correct = True
            else:
                st.error("❌ Incorrect password")
        return False
    return True


# --- MAIN APP ---
if not check_password():
    st.stop()
else:

    st.write("You are logged in! 🎉")

    # --- Streamlit UI ---
    st.title("Deployment Video player")

    key = st.text_input("Provide DropID")
    url = f"media/{key[:16]}/{key[:27]}/{key}"
    if not url.endswith(".mp4"):
        url += ".mp4"

    if st.button("Generate URL and play video"):
        if not url:
            st.error("Please provide the DropID.")
        else:
            ps_url = get_presigned_url(url)
            st.write("Presigned URL (temporary):")
            st.code(ps_url, language="text")

            st.subheader("Video preview")
            st.video(ps_url)
