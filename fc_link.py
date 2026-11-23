"""Connection to the phone over the USB cable (adb forward) or to the simulator. Reconnects by itself."""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
import time

from PySide6.QtCore import QThread, Signal

import fc_proto as P


APP_ID = "com.fatiguecam.sensor"


def apk_path():
    import sys
    here = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, "phone", "FatigueCamSensor.apk"),
              os.path.join(os.path.dirname(sys.executable), "phone", "FatigueCamSensor.apk"),
              os.path.join(os.path.dirname(here), "android", "app", "build", "outputs", "apk", "debug", "app-debug.apk")):
        if os.path.exists(p):
            return p
    return None


def install_app():
    """Install / update the phone app over the cable (grants the camera permission) and start it."""
    apk = apk_path()
    if not apk:
        return False, "קובץ האפליקציה (APK) לא נמצא"
    code, out = adb("install", "-r", "-g", apk, timeout=120)
    if code != 0 or "Success" not in out:
        return False, out[-300:]
    adb("shell", "am", "start", "-n", f"{APP_ID}/.MainActivity")
    return True, "האפליקציה הותקנה והופעלה בטלפון"


def adb_path():
    import sys
    here = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))
    exe_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else here
    for p in (os.path.join(here, "adb", "adb.exe"), os.path.join(exe_dir, "adb", "adb.exe"),
              os.path.join(os.environ.get("LOCALAPPDATA", ""), "Android", "Sdk", "platform-tools", "adb.exe"),
              shutil.which("adb") or ""):
        if p and os.path.exists(p):
            return p
    return None


def adb(*args, timeout=10):
    a = adb_path()
    if not a:
        return 1, "adb not found"
    try:
        r = subprocess.run([a, *args], capture_output=True, text=True, timeout=timeout,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return r.returncode, (r.stdout + r.stderr).strip()
    except Exception as ex:
        return 1, str(ex)


class Link(QThread):
    """Emits every message from the phone as (msg, payload, t_pc_received)."""
    message = Signal(object, object, float)
    status = Signal(str, str)          # level (ok/warn/err), text

    def __init__(self, mode="usb", port=P.PORT, sim_port=8766):
        super().__init__()
        self.mode, self.port, self.sim_port = mode, port, sim_port
        self._sock = None
        self._lock = threading.Lock()
        self._stop = False
        self.connected = False
        self.rtt = float("nan")
        self._ping_id = 0
        self._ping_sent = {}
        self._t_launch = 0.0

    def stop(self):
        self._stop = True
        self._close()

    def send(self, msg, payload=b""):
        with self._lock:
            s = self._sock
            if s is None:
                return False
            try:
                s.sendall(P.pack(msg, payload))
                return True
            except OSError:
                return False

    def _close(self):
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
            self._sock = None
        self.connected = False

    def _connect(self):
        if self.mode == "sim":
            host, port = "127.0.0.1", self.sim_port
        else:
            code, out = adb("devices")
            devs = [ln for ln in out.splitlines()[1:] if ln.strip().endswith("device")]
            if code != 0 or not devs:
                unauth = any("unauthorized" in ln for ln in out.splitlines())
                raise ConnectionError("הטלפון מחובר אך לא אושר – אשר 'ניפוי באגים' במסך הטלפון" if unauth
                                      else "לא נמצא טלפון בכבל (USB debugging כבוי או כבל טעינה בלבד)")
            code, out = adb("forward", f"tcp:{self.port}", f"tcp:{self.port}")
            if code != 0:
                raise ConnectionError(f"adb forward נכשל: {out}")
            host, port = "127.0.0.1", self.port
        s = socket.create_connection((host, port), timeout=3)
        s.settimeout(4)                       # hello comes at once; then metrics many times a second
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with self._lock:
            self._sock = s
        return s

    def run(self):
        last_err = None
        while not self._stop:
            try:
                s = self._connect()
                try:
                    first = P.recv(s)          # "hello" first (an adb forward to nothing closes at once)
                    for _ in range(50):        # a status line may overtake it - skip until hello
                        if first[0].get("type") == "hello":
                            break
                        first = P.recv(s)
                except Exception:
                    if self.mode != "sim" and time.time() - self._t_launch > 10:
                        self._t_launch = time.time()           # the app is not running: start it
                        adb("shell", "am", "start", "-n", f"{APP_ID}/.MainActivity")
                        self.status.emit("warn", "מפעיל את האפליקציה בטלפון...")
                    raise
                s.settimeout(10)
                if first[0].get("type") != "hello":
                    raise ConnectionError("unexpected first message")
                self.connected = True; last_err = None
                self.status.emit("ok", "מחובר")
                self.message.emit(first[0], first[1], time.time())
                pinger = threading.Thread(target=self._pinger, daemon=True); pinger.start()
                while not self._stop:
                    msg, payload = P.recv(s)
                    tr = time.time()
                    if msg.get("type") == "pong" and msg.get("id") in self._ping_sent:
                        self.rtt = tr - self._ping_sent.pop(msg["id"])
                    self.message.emit(msg, payload, tr)
            except Exception as ex:
                was = self.connected
                self._close()
                txt = str(ex) or ex.__class__.__name__
                if was:
                    self.status.emit("err", f"החיבור נותק: {txt}")
                elif txt != last_err:
                    self.status.emit("warn", f"ממתין לטלפון: {txt}")
                last_err = txt
                for _ in range(20):
                    if self._stop:
                        break
                    time.sleep(0.1)

    def _pinger(self):
        while self.connected and not self._stop:
            self._ping_id += 1
            self._ping_sent[self._ping_id] = time.time()
            if len(self._ping_sent) > 20:
                self._ping_sent.pop(min(self._ping_sent))
            self.send({"type": "ping", "id": self._ping_id})
            time.sleep(2.0)
