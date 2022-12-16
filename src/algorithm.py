import numpy as np
import argparse
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from tqdm import tqdm
    
class ImagePalette():
    def __init__(self, args, src):
        # Origin image in RGB
        self.src = np.asarray(src)[..., :3]
        # k in k-means
        self.k = args.k
        # k-means centers in LAB
        self.centers_lab = np.zeros((self.k, 3), dtype=np.float32)
        # Editted k-means centers in LAB
        self.editted_centers_lab = np.zeros((self.k, 3), dtype=np.float32)
        # Editted image centers in LAB
        self.editted_src_lab = np.array([rgb2lab(self.src.reshape(-1, 3) / 255)] * self.k)
        # Grid color in RGB
        self.grid_color = np.mgrid[0:17, 0:17, 0:17].reshape(3, -1).T * 16
        # Editted grid color in LAB
        self.editted_grid_color = np.array([rgb2lab(self.grid_color / 255)] * self.k, dtype=np.float32)
        # Weights of pixels in color grid
        self.linear_weights = np.zeros((self.src.reshape(-1, 3).shape[0], 8))
    
    def reset_palette(self):
        # Reset palette
        self.editted_centers_lab = self.centers_lab.copy()
        self.editted_src_lab = np.array([rgb2lab(self.src.reshape(-1, 3) / 255)] * self.k)
        self.editted_grid_color = np.array([rgb2lab(self.grid_color / 255)] * self.k, dtype=np.float32)

    def get_center_color(self):
        return (lab2rgb(self.editted_centers_lab) * 255).astype(np.uint8)

    def ValidLAB(self, LAB):
        # Check if a LAB pixel is valid
        L, a, b = LAB
        return 0 <= L <= 100 and -128 <= a <= 127 and -128 <= b <= 127

    def LABtoXYZ(self, LAB):
        # convert one lab pixel to xyz
        def f(n):
            return n**3 if n > 6/29 else 3 * ((6/29)**2) * (n - 4/29)

        assert(self.ValidLAB(LAB))

        L, a, b = LAB
        X = 95.047 * f((L+16)/116 + a/500)
        Y = 100.000 * f((L+16)/116)
        Z = 108.883 * f((L+16)/116  - b/200)
        return (X, Y, Z)

    def XYZtoRGB(self, XYZ):
        # convert one xyz pixel to RGB
        def f(n):
            return n*12.92 if n <= 0.0031308 else (n**(1/2.4)) * 1.055 - 0.055

        X, Y, Z = [x/100 for x in XYZ]
        R = f(3.2406*X + -1.5372*Y + -0.4986*Z) * 255
        G = f(-0.9689*X + 1.8758*Y + 0.0415*Z) * 255
        B = f(0.0557*X + -0.2040*Y + 1.0570*Z) * 255
        return (R, G, B)

    def LABtoRGB(self, LAB):
        return self.XYZtoRGB(self.LABtoXYZ(LAB))

    def phi(self, r):
        return np.exp(-r**2 / (2 * self.sigma_r**2))

    def cal_lambda_map(self):
        # Calculate lambda_ij
        sig_r = 0
        for i in range(self.k):
            for j in range(self.k):
                if i == j:
                    continue
                sig_r += np.linalg.norm(self.centers_lab[i] - self.centers_lab[j])
        sig_r /= self.k * (self.k - 1)
        self.sigma_r = sig_r

        matrix_phi = np.zeros((self.k, self.k))
        for i in range(self.k):
            for j in range(self.k):
                matrix_phi[i, j] = self.phi(np.linalg.norm(self.centers_lab[i] - self.centers_lab[j]))
        
        self.lambda_map = np.linalg.inv(matrix_phi)
    
    def cal_pixel_linear_weights(self):    
        # Calculate weights of pixels in color grid    
        pixels = self.src.reshape(-1, 3)
        dif = (pixels % 16) / 16
        weights = np.zeros((pixels.shape[0], 8))
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    weights[:, x * 4 + y * 2 + z] = \
                        (dif[:, 0] * x + (1 - dif[:, 0]) * (1 - x)) * \
                        (dif[:, 1] * y + (1 - dif[:, 1]) * (1 - y)) * \
                        (dif[:, 2] * z + (1 - dif[:, 2]) * (1 - z))

        self.linear_weights = weights    

    def cal_w_matrix(self):
        # Calculate wi for each patelle
        grid_color_lab = rgb2lab(self.grid_color / 255)
        w = np.zeros(self.editted_grid_color.shape[:2])
        for i in range(self.k):
            for j in range(self.k):
                lambda_ij = self.lambda_map[i, j]
                phi = self.phi(np.linalg.norm(grid_color_lab - self.centers_lab[j], axis=1))
                w[i] += lambda_ij * phi

        # Clamp w
        w = np.clip(w, 0, None)
        # Normalize w
        w = w / np.sum(w, axis=0)
        # Reshape w to conduct matrix multiplication 
        self.w = w.reshape(w.shape[0], w.shape[1], 1)

    def select_init_center(self, average_labs, counts, sigma_a=80):
        '''
        An optimization to select the initial center
        '''
        centers = np.zeros((self.k, 3))
        
        # Initial weight is the number of pixels in the bin
        weights = counts.astype(np.float32)

        for i in range(len(centers)):
            # Select the center by the maximum weight
            temp_c = average_labs[weights.reshape(-1).argmax()]
            centers[i] = temp_c
            # Update weights
            dists = np.linalg.norm(average_labs - temp_c, axis=1)
            weights *= (1 - np.exp(-dists ** 2 / sigma_a ** 2))
        
        # Append all black center
        centers = np.concatenate((centers, rgb2lab(np.zeros((1, 3)))))
        return centers

    def kmeans(self):
        # Devide 256^3 RGB space to 16^3 bins
        average_rgbs = np.zeros((16, 16, 16, 3))
        counts = np.zeros((16, 16, 16), dtype=np.int32)

        # Initialize bins
        for rgb in tqdm(self.src.reshape(-1, 3)):
            average_rgbs[rgb[0] >> 4, rgb[1] >> 4, rgb[2] >> 4] += rgb
            counts[rgb[0] >> 4, rgb[1] >> 4, rgb[2] >> 4] += 1
        
        # Delete empty bins
        average_rgbs = average_rgbs[counts > 0]
        counts = counts[counts > 0]
        
        average_rgbs /= counts.reshape(-1, 1)
        average_labs = rgb2lab(average_rgbs / 255)

        centers = self.select_init_center(average_labs, counts)
        new_centers = np.zeros_like(centers)

        # K-means
        while True:
            # Assign pixels to the nearest center
            dists = np.linalg.norm(average_labs.reshape(-1, 1, 3) - centers.reshape(1, -1, 3), axis=2)
            labels = dists.argmin(axis=1)
            # Update centers
            for i in range(len(centers) - 1):
                new_centers[i] = np.average(average_labs[labels == i], axis=0, weights=counts[labels == i])
            if np.all(centers == new_centers):
                break
            centers = new_centers

        # Sort by centers' first dim
        self.centers_lab = centers[np.argsort(centers[:, 0])][1:]
        self.editted_centers_lab = self.centers_lab.copy()

        # Calculate lambda map 
        self.cal_lambda_map()
        # Calculate pixel linear weights
        self.cal_pixel_linear_weights()
        # Calculate w matrix
        self.cal_w_matrix()

        # Convert centers to RGB uint8
        return (lab2rgb(self.editted_centers_lab) * 255).astype(np.uint8)

    def get_img(self):
        return self.src

    def out_boundary(self, test_lab):
        if not self.ValidLAB(test_lab):
            return True
        test_rgb = self.LABtoRGB(test_lab)
        out_threshold = 0.5
        return test_rgb[0] < -out_threshold or test_rgb[0] > 255 + out_threshold or \
            test_rgb[1] < -out_threshold or test_rgb[1] > 255 + out_threshold or \
            test_rgb[2] < -out_threshold or test_rgb[2] > 255 + out_threshold

    def find_boundary(self, vsrc, dir, l=0.0, r=10.0):
        # Let r big enough to reach boundary
        while not self.out_boundary(vsrc + r * dir):
            r *= 2
        for i in range(10):
            mid = (l + r) // 2
            if self.out_boundary(vsrc + mid * dir):
                r = mid
            else:
                l = mid
    
            if r - l <= 0.01:
                break
        return vsrc + l * dir

    def update_image(self, new_color, pid):
        edit_grid_lab = rgb2lab(self.grid_color / 255)

        # Calculate t
        centers_l = np.concatenate((np.asarray([0. - 1e-1]), self.centers_lab[:, 0], np.asarray([100. + 1])))
        inds = np.digitize(edit_grid_lab[:, 0], centers_l) - 1
        t = (edit_grid_lab[:, 0] - centers_l[inds]) / (centers_l[inds + 1] - centers_l[inds])

        # Recover the other center's L when it isn't editted
        for i in range(self.k):
            if np.all(self.editted_centers_lab[i, 1:] == self.centers_lab[i, 1:]):
                self.editted_centers_lab[i] = self.centers_lab[i]

        # Update palette
        editted_centers_l = np.concatenate((np.asarray([0. - 1e-3]), self.editted_centers_lab[:, 0], np.asarray([100. + 1e-3])))
        editted_centers_l[pid + 1] = rgb2lab(new_color / 255)[0]
        for i in range(0, pid + 1):
            if editted_centers_l[i] > editted_centers_l[pid + 1]:
                editted_centers_l[i] = editted_centers_l[pid + 1]
        for i in range(pid + 2, self.k + 1):
            if editted_centers_l[i] < editted_centers_l[pid + 1]:
                editted_centers_l[i] = editted_centers_l[pid + 1]

        self.editted_centers_lab[pid] = rgb2lab(new_color / 255)
        self.editted_centers_lab[:, 0] = editted_centers_l[1:-1]

        # Edit L in LAB space
        edit_grid_lab[:, 0] = editted_centers_l[inds] + t * (editted_centers_l[inds + 1] - editted_centers_l[inds])

        # Edit A B in LAB space
        print("Calculating color for each grid...")
        C = self.centers_lab[pid]
        C_ = self.editted_centers_lab[pid]
        Cb = self.find_boundary(C, C_ - C)

        xs = np.zeros(edit_grid_lab.shape)
        x0s = np.zeros_like(xs)
        xbs = np.zeros_like(xs)
        for color, i in tqdm(zip(edit_grid_lab, range(len(xs)))):
            xs[i] = color
            x0s[i] = xs[i] + C_ - C
            if self.out_boundary(x0s[i]):
                xbs[i] = self.find_boundary(C_, x0s[i] - C_, 0, 1)
            else:
                xbs[i] = self.find_boundary(xs[i], x0s[i] - xs[i])

        edit_grid_lab[:, 1:] = (xs + (xbs - xs) * np.linalg.norm(C_ - C) * np.clip(np.linalg.norm(xbs - xs, axis=1)[:, np.newaxis] / np.linalg.norm(Cb - C), None, 1) / np.linalg.norm(xbs - xs, axis=1)[:, np.newaxis])[:, 1:]
        self.editted_grid_color[pid] = edit_grid_lab
  
        # Get final color
        final_grid_color = np.sum(self.w * self.editted_grid_color, axis=0)
        final_grid_color = lab2rgb(final_grid_color) * 255

        # Prepare for update pixel color
        grid_selected = np.zeros((self.src.reshape(-1, 3).shape[0], 8, 3))
        grid_selected_id = np.arange(8)
        edit_grid_id = np.zeros(8, dtype=np.int)
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    edit_grid_id[x * 2**2 + y * 2 + z] = x * 17**2 + y * 17 + z

        # Calculate color for each grid by interpolation
        print('Calculating color for all pixels by interpolation...')
        editted_rgb = np.zeros_like(self.src.reshape(-1, 3))
        for pixel, i in tqdm(zip(self.src.reshape(-1, 3), range(self.src.reshape(-1, 3).shape[0]))):
            grid_pos = pixel // 16
            grid_id = grid_pos[0] * 17**2 + grid_pos[1] * 17 + grid_pos[2]
            
            grid_selected[i][grid_selected_id] = \
                final_grid_color[grid_id + edit_grid_id]
            editted_rgb[i] = np.dot(self.linear_weights[i], grid_selected[i])

        return editted_rgb.reshape(self.src.shape)

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="K-means clustering")
    parser.add_argument(
        '--image',
        type=str, 
        help='Path to the image',
        default='../data/src1.png'
    )
    parser.add_argument(
        '--k', 
        type=int, 
        help='Number of clusters',
        default=5
    )
    args = parser.parse_args()

    # Read image, ignoring alpha channel
    src = Image.open(args.image)
    palette = ImagePalette(args, src)
    # Get centers by kmeans
    centers = palette.kmeans()

    
    for i in range(len(centers)):
        color_block = np.array([centers[i]] * 10000).reshape(100, 100, 3).astype(np.uint8)
        Image.fromarray(color_block).save(f'data/center{i}.png')

