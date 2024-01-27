import torch

def calculate_angle(tensor1, tensor2):
    # Flatten tensors to vectors
    vector1 = tensor1
    vector2 = tensor2

    # Calculate dot product
    dot_product = torch.mm(vector1, vector2)

    # Calculate magnitudes
    magnitude1 = torch.norm(vector1)
    magnitude2 = torch.norm(vector2)

    # Calculate cosine of the angle
    cosine_angle = dot_product / (magnitude1 * magnitude2)

    # Calculate the angle in radians
    angle_rad = torch.acos(torch.clamp(cosine_angle, -1.0, 1.0))
    
    angle_rad = torch.tensor(angle_rad)

    # Convert angle to degrees
    angle_deg = torch.rad2deg(angle_rad)

    return angle_rad  # Convert to Python float

# Example usage:
tensor1 = torch.randn(100, 10)
tensor2 = torch.randn(10, 20)

angle_between_tensors = calculate_angle(tensor1, tensor2)
print("Angle between tensors:", angle_between_tensors, "degrees")
