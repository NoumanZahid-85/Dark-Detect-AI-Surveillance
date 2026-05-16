import cv2
import sys
from pathlib import Path

# Add project root to path if needed, but normally running from root works
sys.path.append(str(Path(__file__).parent.parent))

from utils.enhancement import enhance_image
from utils.augmentation import simulate_low_light

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_enhancement.py <path_to_image>")
        print("Example: python scripts/test_enhancement.py data/ExDark/images/test/2015_00001.jpg")
        sys.exit(1)
        
    img_path = sys.argv[1]
    if not Path(img_path).exists():
        print(f"File not found: {img_path}")
        sys.exit(1)
        
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load image. It might be corrupt or an unsupported format.")
        sys.exit(1)
        
    print(f"Loaded image {img_path} with shape: {img.shape}")
    
    print("Simulating low light...")
    simulated = simulate_low_light(img, severity='medium')
    
    print("Applying enhancement (CLAHE + Gamma)...")
    enhanced = enhance_image(simulated, method='both')
    
    # Save output to view since we might not have a GUI environment
    out_dir = Path("results/test")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(out_dir / "1_original.jpg"), img)
    cv2.imwrite(str(out_dir / "2_simulated.jpg"), simulated)
    cv2.imwrite(str(out_dir / "3_enhanced.jpg"), enhanced)
    
    print(f"Saved results to {out_dir}/. Check the images to verify visual improvements.")

if __name__ == "__main__":
    main()
