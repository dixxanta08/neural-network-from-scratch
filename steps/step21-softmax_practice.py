import numpy as np

# softmax_output=[0.7,0.1,0.2] 
# softmax_output = np.array(softmax_output).reshape(-1,1)
# print(softmax_output)
# print(softmax_output.shape)
# print(np.eye(softmax_output.shape[0]))
# # print(softmax_output *np.eye(softmax_output.shape[0]))

# print(np.diagflat(softmax_output) - softmax_output* softmax_output.T)

y_pred = np.array([
    [0.70, 0.20, 0.10],   # Sample 0
    [0.10, 0.80, 0.10],   # Sample 1
    [0.20, 0.30, 0.50]    # Sample 2
])
samples = len(y_pred)      # 3
y_true = np.array([0, 1, 2])
print(y_pred[
                range(samples),
                y_true
            ])
