"""Launch the completed expanded classifier locally."""
import webbrowser
import threading
from ecosort_runtime import application_running, launch

if __name__ == "__main__":
    if application_running():
        webbrowser.open("http://127.0.0.1:7861")
        raise SystemExit(0)
    demo = launch(prevent_thread_lock=True)
    webbrowser.open("http://127.0.0.1:7861")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        demo.close()
