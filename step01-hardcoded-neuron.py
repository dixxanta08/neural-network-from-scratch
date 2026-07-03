# Step 1: Hardcode a single neuron (3 inputs → 1 output)



# price of house, location (distance from school) and age
# price = 40000
# location = 25
# age = 10

# scale or normalization
price = 0.9
location = 0.3
age = 0.2

weight_price  = 0.6
weight_location = 0.8
weight_age = 0.3

output = price*weight_price + location*weight_location + age* weight_age
print(output)

# The neuron simply computes a weighted sum of its inputs.
# The meaning of this score depends entirely on the problem we are solving.
# By itself, 0.84 has no meaning.
# It only becomes meaningful when we define a task (e.g., house price,
# loan approval, spam detection) and train the network using labeled data.