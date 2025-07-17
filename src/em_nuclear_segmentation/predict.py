from em_nuclear_segmentation.utils.predict_utils import predict

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: predict-nuclei path/to/image.tif")
        sys.exit(1)
    predict(sys.argv[1])
