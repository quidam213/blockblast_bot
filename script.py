import subprocess
import cv2
import numpy as np

class Board:
    def __init__(self, x : int = 58, y : int = 580, w : int = 968, h : int = 968):
        self.x : int = x;
        self.y : int = y;
        self.w : int = w;
        self.h : int = h;

    def crop(self, img):
        return img[self.y:self.y+self.h, self.x:self.x+self.w];

    def get_binary(self, img, tolerance : int = 50):
        board_img = self.crop(img);
        gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY);
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV);
        return binary;

    def get_board_state(self, img, grid_size=8):
        binary_img = self.get_binary(img);
        h, w = binary_img.shape
        cell_w = w // grid_size
        cell_h = h // grid_size
        board = []
        for j in range(grid_size):
            row = []
            for i in range(grid_size):
                x0, y0 = i * cell_w, j * cell_h
                cell = binary_img[y0:y0+cell_h, x0:x0+cell_w]
                fill_ratio = np.mean(cell) / 255
                row.append(1 if fill_ratio > 0.3 else 0)  # seuil ajustable
            board.append(row)
        return np.array(board)

def capture_screen():
    result = subprocess.run(
        ["adb", "exec-out", "screencap", "-p"],
        stdout=subprocess.PIPE
    )
    img = cv2.imdecode(np.frombuffer(result.stdout, np.uint8), cv2.IMREAD_COLOR);
    return img;

def main():
    board : Board = Board();
    while True:
        # Capture l'écran + crop le board
        img = capture_screen();
        board_img = board.crop(img);
        binary = board.get_binary(img);

        board_state = board.get_board_state(img);
        print(f"State du board :\n{board_state}\n\n");

        # Show les images (écran + board)
        cv2.imshow("Full Capture", img);
        cv2.imshow("Board Only", board_img);
        cv2.imshow("Binary Board", binary);

        # Close avec key 'Q'
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break;

    cv2.destroyAllWindows();

if __name__ == "__main__":
    main();

