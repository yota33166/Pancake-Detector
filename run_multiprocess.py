from multiprocessing import Process, Queue

from detect_pancake import PancakeDetector
import servo_controller.servo_control as servo_control
import logging


def run_camera(cmd_queue: Queue) -> None:
    detector = PancakeDetector(
        camera_index=0,
        frame_width=480,
        frame_height=320,
        fps=20,
        serial_port='ttyUSB0',
        command_queue=cmd_queue,
        trigger_cooldown_s=3,
        max_pour_time_s=2.0,
    )
    detector.run()


def run_servos(cmd_queue: Queue) -> None:
    controller = servo_control.ServoController(command_queue=cmd_queue)
    controller.run()


if __name__ == "__main__":
    command_queue: Queue = Queue()

    cam_proc = Process(target=run_camera, args=(command_queue,), daemon=True)
    servo_proc = Process(target=run_servos, args=(command_queue,), daemon=True)

    cam_proc.start()
    servo_proc.start()

    cam_proc.join()
    logging.debug("camera process exited: exitcode=%s", cam_proc.exitcode)
    if cam_proc.exitcode not in (0, None):
        logging.debug("camera died early; sending sentinel to servos")
        command_queue.put(None)

    servo_proc.join()
