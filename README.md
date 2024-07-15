Methodology:

➔ All the questions in the dataset has single answers, so the ideology is to treat the
problem as multi-class classification task.

➔ calculated all the unique answers in from the train, validation and test datasets
which turned out to be 4879 and each answer is treated as an individual class.

➔ For each input of image and its corresponding question, the hybrid model generates
4879 outputs, take the index of maximum logit for each sample, which is the
predicted class.

➔ The predicted answer is compared with the actual answer to calculate the loss for
back propogation.

➔ Accuracy is calculated by comparing the predicted answer with the actual answer.

Overview of Model
➔ Image Encoding (Enhanced ResNet):
    ◼ Loading a pretrained ResNet – 101 Model
    
    ◼ Removed last two layers of ResNet to get feature map instead of classification
    
    ◼ Added two more convolution layers with batch normalization and adaptiveaverage pooling with Relu activation function.
    
        - nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1): 2D convolutional layer reduces channels from 2048 to 1024
        
        - nn.BatchNorm2d(1024): Batch normalization for 1024 channels.
        
        - nn.ReLU(): ReLU activation function.
        
        - nn.AdaptiveAvgPool2d((7, 7)): Adaptive average pooling to (7, 7).
        
        - nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1): 2D convolutional layer reducing channels from 1024 to 512.
        
        - nn.BatchNorm2d(512): Batch normalization for 512 channels.
        
        - nn.ReLU(): ReLU activation function.
        
        - nn.AdaptiveAvgPool2d((1, 1)): Adaptive average pooling to (1, 1).
        
    ◼ self.fc = nn.Linear(512, 768): Fully connected layer transforming 512 channels to 768 dimensions.
    
→ Forward Method:

    - Images are fed to ResNet model
    
    - Feature maps from ResNet are fed to additional convolution layers, to reduce the dimensions and applied adaptive average pooling
    
    - To get an embedding size of 768, the results from the above convolutional layers are flattened and passed to fully connected layer


➔ Text Encoding:
    ◼ A pretrained BERT model is used to extract the text features.


➔ Fusion Transformer:

◼ def __init__(self, emb_size, num_heads, num_layers, num_classes): Initializing the vision transformer with embedding size, number of heads, number of layers and number of classes to predict.
    
◼ self.pos_embedding = nn.Parameter(torch.randn(1, 2, emb_size)): Defining a positional embedding as a learnable parameter.

◼ self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size)): Defineing a classification token as a learnable parameter.

◼ self.transformer = nn.Transformer(d_model=emb_size, nhead=num_heads, num_encoder_layers=num_layers, batch_first=True): Initializing the Transformer model.

    - d_model=emb_size: Dimension of the input embeddings.
    
    - nhead=num_heads: Number of attention heads.
    
    - num_encoder_layers=num_layers: Number of encoder layers.
    
    - batch_first=True: to specify the input/output tensors will have batch size as the first dimension.
    
    - self.fc = nn.Linear(emb_size, num_classes): Fully connected layer to map the transformer output to the number of classes.

→ Forward Method:

    ◼ batch_size = x.size(0): Gets the batch size.
    
    ◼ cls_tokens = self.cls_token.expand(batch_size, -1, -1): Expandind the classification token to match the size of batches.
    
    ◼ x = torch.cat((cls_tokens, x), dim=1): Concatenating the classification token with the input.
    
    ◼ x += self.pos_embedding: Adding positional embedding to the input.
    
    ◼ x = self.transformer(x, x): Passing the input to the Transformer model.
    
    ◼ x = self.fc(x[:, 0]): Using the output based on classification token and passes it to the fully connected layer.
    
    ◼ Returning the final output




➔ Hybrid Model:

◼ Enhanced ResNet for Image Encoding

◼ BERT Model for Text Encoding (questions)

◼ Image features and Question features are concatenated and transformed using a fully connected layer

◼ The transformed features are reshaped to match the input size of fusion Transformer.

◼ Important parts in the combined features are enhanced using attention layer.

◼ The output of attention layer is further processed by one more fully connected layer.

◼ The attention processed features are passed to fusion Transformer for final classification.


Results:

![image](https://github.com/user-attachments/assets/39b0826e-a1de-4c48-8db8-a8bf14ab9d63)


The model achieved an accuracy of 55.7% on the unfiltered data, which is commendable given the
dataset's complexity involving 4879 unique classes. For the ‘Yes/No’ question-answer pairs, the
model yielded an accuracy of 86%. This is a significant improvement over the original author's
results in the PathVQA paper, where the highest accuracy was 68% using a combination of GRU
and Faster R-CNN models. The table below shows accuracies from PathVQA paper.

![image](https://github.com/user-attachments/assets/87c48efe-cf52-45c9-85dd-2f2218b6ece5)



