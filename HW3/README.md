# CSCE636_Deep_Learning


You will use the Python programming language and PyTorch for this assignment.


Coding Task 
-----------------------
**GPT for text generation tasks**: In this assignment, you will
implement a decoder-only Transformer model on the SCAN dataset using PyTorch. The goal
of SCAN is to translate commands presented in simplified natural language into a sequence
of actions. In this translation task, the generation model will take a command sentence as
input and output the corresponding action sequence. For more details, please refer to the
dataset available at https://huggingface.co/datasets/scan. You can also refer to its
original paper at https://arxiv.org/pdf/1711.00350.pdf for the introduction of the task.
The starting code is provided in the ”code” folder. All the architecture and training code is
provided, but you need to implement the CSABlock (causal self-attention blcok) class in the
”model.py” file and the generate sample function in the ”generate.py” file. Similarly, in this
assignment, you must use a GPU. 

Requirements are the same as our project including Python, NumPy, PyTorch, plus tqdm
and datasets (from Hugging Face). Other packages for transformer implementations are not
allowed.

Please submit running report (briefly explain how you complete those functions, capture
screen shots of your training/validation loss, test acc etc., anything required in the original
assignment) for all coding tasks. And paste training and testing console record as an appendix
in your report. You mustn’t submit the pre-trained model and the dataset which could be
potentially large. Code should be written in a clean and organized way. Comments in the code
are required. Questions that only include code without a report and explanations will not be
graded. Please make sure all required files are included in your submission by downloading
your submission on Canvas and double check it. We will not accept any late submission or
any re-submission.


(a) (15 points) Run the starting code directly to download the SCAN dataset automatically.
Read the code, understand the data processing, and answer the following questions:
What is a tokenizer? How does a tokenizer process the raw data? What is the size of
the vocabulary?

(b) (5 points) What is the maximum length of the input sequence? How should we determine
the maximum length of the input sequence?

(c) (30 points) Implement the CSABlock class in the ”model.py” file. Which steps are
involved in the self-attention mechanism? Which step is critical to make it ”causal” in
your code? Why do we need a mask in the forward function of the class ”GPT”? Report
your training process and results.

(d) (15 points) Implement the generate sample function in the ”generate.py” file. What
is the generation process? Please explain the process using a concrete example in the
dataset.

(e) (15 points) Re-train your GPT model using different number of layers, number of heads,
and number of embeddings. Report your validation loss, time per epoch, and test results
in a table. What is the impact of these hyperparameters on the model performance?

(f) (20 points) There are other splits (instead of the ”simple” one) of the SCAN dataset
https://huggingface.co/datasets/scan. You can use other splits by simply setting
the CLI argument ”data split” to the names of the splits. Please try to use another split
(your choice) and compare your result with the results reported in the original paper
https://arxiv.org/pdf/1711.00350.pdf. What is the split you choose? What is the
type of evaluation that the split is designed for? What insights do you get from the
comparison? 



