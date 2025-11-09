import subprocess
import cv2
import numpy as np

class Element:
    def __init__(self, x : int, y : int, w : int, h : int):
        self.x : int = x;
        self.y : int = y;
        self.w : int = w;
        self.h : int = h;

    def crop(self, img):
        return img[self.y:self.y+self.h, self.x:self.x+self.w];

    def get_binary(self, img, tolerance : int):
        crop_img = self.crop(img);
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY);
        _, binary = cv2.threshold(gray, tolerance, 255, cv2.THRESH_BINARY_INV);
        return binary;

class Board(Element):
    def __init__(self, x : int = 58, y : int = 580, w : int = 968, h : int = 968):
        super().__init__(x, y, w, h);

    def get_binary(self, img, tolerance : int = 50):
        return super().get_binary(img, tolerance);

    def get_board_state(self, img, grid_size : int = 8):
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
                row.append(0 if fill_ratio > 0.3 else 1)  # seuil ajustable
            board.append(row)
        return np.array(board)

class Pieces(Element):
    def __init__(self, x : int = 58, y : int = 1690, w : int = 968, h : int = 275):
        super().__init__(x, y, w, h);

    def get_binary(self, img, tolerance : int = 90):
        crop_img = self.crop(img);
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY);
        _, binary = cv2.threshold(gray, tolerance, 255, cv2.THRESH_BINARY);
        return binary;

    def get_pieces(self, img, pcount : int = 3, grid_size : int = 5):
        binary_img = self.get_binary(img);
        h, w = binary_img.shape;
        piece_w = w // pcount;
        pieces = [];
        for i in range(pcount):
            x0 = i * piece_w;
            piece_crop = binary_img[0:h, x0:x0+piece_w];
            matrix = self._to_matrix(piece_crop, grid_size);
            pieces.append(matrix);
        return pieces;

    def _to_matrix(self, piece_img, grid_size : int = 5):
        h, w = piece_img.shape;
        cell_h = h // grid_size;
        cell_w = w // grid_size;
        mat = [];
        for j in range(grid_size):
            row = [];
            for i in range(grid_size):
                x0, y0 = i * cell_w, j * cell_h;
                cell = piece_img[y0:y0+cell_h, x0:x0+cell_w];
                fill_ratio = np.mean(cell) / 255;
                row.append(1 if fill_ratio > 0.3 else 0);
            mat.append(row);
        return np.array(mat);

def capture_screen():
    result = subprocess.run(
        ["adb", "exec-out", "screencap", "-p"],
        stdout=subprocess.PIPE
    )
    img = cv2.imdecode(np.frombuffer(result.stdout, np.uint8), cv2.IMREAD_COLOR);
    return img;

def main():
    board : Board = Board();
    pieces : Pieces = Pieces();
    while True:
        # Capture l'écran + crop le board
        img = capture_screen();
        board_img = board.crop(img);
        binary_board = board.get_binary(img);
        pieces_img = pieces.crop(img);
        binary_pieces = pieces.get_binary(img);

        board_state = board.get_board_state(img);
        print(f"State du board :\n{board_state}\n");
        
        pieces_state = pieces.get_pieces(img);
        print("\nPièces :");
        for i, p in enumerate(pieces_state):
            print(f"\nP{i+1} :\n{p}");
        print("\n\n");

        # Show les images (écran + board)
        #cv2.imshow("Full Capture", img);
        #cv2.imshow("Board Only", board_img);
        #cv2.imshow("Binary Board", binary_board);
        cv2.imshow("Pieces only", pieces_img);
        cv2.imshow("Binary pieces", binary_pieces);

        # Close avec key 'Q'
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break;

    cv2.destroyAllWindows();

if __name__ == "__main__":
    main();

