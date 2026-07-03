# Move inputs and weights into vectors (lists/arrays)

# Previously, we stored each input and weight in separate variables.
# Now we group related values into lists (vectors), making the code
# easier to scale as the number of features grows.
#
# In the next step, replace Python lists with NumPy arrays

# Inputs represent the features describing a single house.
# In this example:
# inputs = [price, location, age]

inputs = [0.9, 0.3, 0.2]
weights = [0.6, 0.8, 0.3]

bias = 0.5

print("Paired inputs and weights:", list(zip(inputs, weights)))

for feature, weight in zip(inputs, weights):
    print(f"Input: {feature} | Weight: {weight}")

products = [feature * weight for feature, weight in zip(inputs, weights)]
print("Individual products (Input × Weight):", products)

dot_product = sum(products)
print("Final weighted sum (Dot Product):", dot_product)

output = dot_product + bias
print("Output:", output)