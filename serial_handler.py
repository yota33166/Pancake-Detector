import serial
import serial.tools.list_ports
import time
import threading

class SerialHandler:
    def __init__(self, port="/dev/ttyACM0", baudrate=115200, timeout=1):
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2)  # 接続安定のため少し待つ
        self.running: bool = False
        self.thread: threading.Thread = None
        self.callback: callable = None

    @staticmethod
    def list_ports():
        """利用可能なシリアルポート一覧を返す"""
        return [(p.device, p.description) for p in serial.tools.list_ports.comports()]

    def send(self, data: str):
        """Picoに文字列を送信"""
        if self.ser.is_open:
            self.ser.write((data + "\r\n").encode("utf-8"))

    def start_listening(self, callback):
        """受信を別スレッドで待ち受け。callback関数に受信文字列を渡す"""
        self.callback = callback
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()

    def _listen_loop(self):
        while self.running:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode("utf-8").rstrip()
                if self.callback:
                    self.callback(line)
            else:
                time.sleep(0.01)  # ポーリング間隔を抑制
                    
    def stop(self):
        """受信スレッドを止める"""
        self.running = False
        if self.thread:
            self.thread.join()

    def close(self):
        """ポートを閉じる"""
        self.stop()
        if self.ser.is_open:
            self.ser.close()


if __name__ == "__main__":
    def print_received(data):
        print(f"Received: {data}")

    serial_handler = SerialHandler(port="COM3")  # Windowsの場合
    # serial_handler = SerialHandler(port="/dev/ttyACM0")  # Linuxの場合
    serial_handler.start_listening(print_received)

    try:
        while True:
            user_input = input("Send to Pico (or 'exit' to quit): ")
            if user_input.lower() == 'exit':
                break
            serial_handler.send(user_input)
    except KeyboardInterrupt:
        pass
    finally:
        serial_handler.close()
