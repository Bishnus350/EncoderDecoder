# EncoderDecoder
- Encoding and Decoding Image.
- Why do we need to encode and decode images?
- - The main answer is to make our computer efficient and able to handle large volumes of data. An image from an iPhone is like 10 MB and from a good DSLR is also nearly 10 MB. When we are doing Image processing for a hundred thousand, the data volume will be significantly high. Processing high volume is computationally expensive. We may need HIgh computing GPUs and it is very expensive. Here are the Encoder and decoder functions.
  - Encoding functions decrease the size of the image by reducing features and preserving important features needed for image classification. Later, the distorted image will be reconstructed again and the volume of the image is less and it is easy to process for image classifications.
