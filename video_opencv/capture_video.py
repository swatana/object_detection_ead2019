import cv2
from datetime import datetime
from argparse import ArgumentParser

def main(args):
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_dir = "./videos/" + args.category + '/'
    filename = datetime.now().strftime("%Y-%m%d-%H-%M-%S") + '.avi' 
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter(output_dir + filename, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        writer.write(frame)

    writer.release()
    cap.release()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--category', type=str, default='others', help = 'category of video')
    main(parser.parse_args())
