from flask import Flask, render_template, request
import subprocess, os, signal, json, time, webbrowser, threading, sys
import requests

app = Flask(__name__)

BASE_PORT = 5001
PORT_FILE = "bot_ports.json"
BOT_DIR = "bots"
running_process = None

# Load saved ports
if os.path.exists(PORT_FILE):
    with open(PORT_FILE, "r") as f:
        BOT_PORTS = json.load(f)
else:
    BOT_PORTS = {}

def list_bots():
    return {
        folder: os.path.join(BOT_DIR, folder)
        for folder in os.listdir(BOT_DIR)
        if os.path.isdir(os.path.join(BOT_DIR, folder)) and os.path.exists(os.path.join(BOT_DIR, folder, "app.py"))
    }

@app.route("/", methods=["GET", "POST"])
def index():
    bots = list_bots()
    statuses = get_bot_statuses(bots)

    if request.method == "POST":
        action = request.form.get("action")
        selected = request.form.get("bot")

        if action == "stop_all":
            kill_running_bot()
            return render_template("index.html", bots=bots, status=get_bot_statuses(bots))

        if selected:
            bot_path = bots[selected]
            port = BOT_PORTS.get(selected, get_next_port())
            BOT_PORTS[selected] = port

            with open(PORT_FILE, "w") as f:
                json.dump(BOT_PORTS, f)

            kill_running_bot()
            launch_bot(bot_path, port)
            wait_for_bot(port)  # Wait until bot is ready
            webbrowser.open_new_tab(f"http://127.0.0.1:{port}")
            return render_template("index.html", bots=bots, status=get_bot_statuses(bots))

    return render_template("index.html", bots=bots, status=statuses)

def get_next_port():
    used_ports = set(BOT_PORTS.values())
    port = BASE_PORT
    while port in used_ports:
        port += 1
    return port

def launch_bot(bot_dir, port):
    global running_process
    full_path = os.path.abspath(bot_dir)
    script_path = os.path.join(full_path, "app.py")
    print(f"🚀 Launching bot at {script_path} on port {port}")

    try:
        running_process = subprocess.Popen(
            [sys.executable, script_path, str(port)],
            cwd=full_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait for 5 seconds and check if it exited early
        time.sleep(5)

        if running_process.poll() is not None:
            stdout, stderr = running_process.communicate()
            print("📤 STDOUT:\n", stdout)
            print("❌ STDERR:\n", stderr)
        else:
            print("✅ Bot subprocess appears to be running.")
    except Exception as e:
        print(f"❌ Exception while launching bot: {e}")



        # Stream logs in real time
        time.sleep(2)
        if running_process.poll() is not None:
            err = running_process.stderr.read().decode()
            print("❗ Error from subprocess:\n", err)
        else:
            print("✅ Bot subprocess started successfully.")

    except Exception as e:
        print(f"❌ Exception while launching bot: {e}")

        stdout, stderr = running_process.communicate(timeout=5)
        print("🔧 Bot STDOUT:\n", stdout.decode())
        print("❌ Bot STDERR:\n", stderr.decode())

        # Wait briefly to let it start and capture any errors
        time.sleep(1)
        if running_process.poll() is not None:
            err = running_process.stderr.read().decode()
            print("❗ Error from subprocess:\n", err)

    except Exception as e:
        print(f"❌ Exception while launching bot: {e}")


def wait_for_bot(port, timeout=10):
    url = f"http://127.0.0.1:{port}"
    for _ in range(timeout * 4):  # check every 0.25s
        try:
            requests.get(url)
            return True
        except:
            time.sleep(0.25)
    return False


def kill_running_bot():
    global running_process
    if running_process and running_process.poll() is None:
        try:
            print("Killing previous bot...")
            os.kill(running_process.pid, signal.SIGTERM)
        except:
            pass
        running_process = None

def get_bot_statuses(bots):
    statuses = {}
    for name, path in bots.items():
        port = BOT_PORTS.get(name)
        if not port:
            statuses[name] = "🔴 Not Launched"
            continue
        try:
            requests.get(f"http://127.0.0.1:{port}", timeout=0.5)
            statuses[name] = f"🟢 Running on port {port}"
        except:
            statuses[name] = "🔴 Stopped"
    return statuses

if __name__ == "__main__":
    def open_browser():
        webbrowser.open_new_tab("http://127.0.0.1:7000")

    threading.Timer(1.5, open_browser).start()
    app.run(port=7000, debug=True)
