# Initialize variables for calculating the sum and count of values in the 11th column
column_sum = 0.0
count = 0

filepath = "/home/ircvlab/VIO/odometry.txt"
# filepath = "/home/ircvlab/VIO/comparison_odometry.txt"

# Open the file in read mode
with open(filepath, "r") as file:
    # Iterate through each line in the file
    for line in file:
        # Split the line by spaces to get individual values
        values = line.split()
        # Check if there are enough columns (at least 11)
        if len(values) >= 11:
            # Convert the 11th column value to float and add it to the sum
            column_sum += float(values[10])
            # Increment the count
            count += 1

# Calculate the average if there are values in the column
average = column_sum / count if count > 0 else 0

print("Average of the 11th column:", average)
