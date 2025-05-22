        for data in positions:
            if len(data) >= 4:  # Ensure at least 4 values exist
                px, py, cropped_w, cropped_h = data[:4]  # Extract the first 4 values
                print(f"px: {px}, py: {py}, width: {cropped_w}, height: {cropped_h}")
            else:
    