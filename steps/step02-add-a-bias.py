# Add a bias term

# a bias is a universal rule that applies to all inputs equally.
# a fixed offset that shifts the neuron’s output up or down

price = 0.9
location = 0.3
age = 0.2

weight_price  = 0.6
weight_location = 0.8
weight_age = 0.3

bias = 0.5
# bias is 0.5 which means 
# “Even if inputs are small or zero, the neuron is slightly more likely to output a higher value.”


# logit or sum or weight_sum
weighted_sum = price*weight_price + location*weight_location + age* weight_age

output = weighted_sum + bias

print(output)

# Output is 1.34
# This is a raw score (logit), not a decision
# It indicates the current weighted evidence is positive
# but has no real-world meaning until we define a task + train the model