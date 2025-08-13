# DCGAN_EXP
>A simple PyTorch implementation of DCGAN, a variant of Generative Adversial Networks on the FashionMNIST images.

## Generated images after last epoch
The model is trained on 60,000 train-data images of the FashionMNIST dataset for 50 epochs on 2 T4 GPUs.
Training images are 28x28 by default and they were resized to 32x32 for this experiment.
Since the images have low resolution,the generator is *updated twice* every step during training.



<img width="634" height="633" alt="Screenshot 2025-07-31 at 8 20 49 PM" src="https://github.com/user-attachments/assets/a3753e27-7347-46d7-8d57-ee2a8e7485b4" />



## Generator progression over all epochs



![G progression gif](https://github.com/user-attachments/assets/29311415-043e-4dd9-a3e2-d3521d6d9042)


## Real images and fake images



<img width="1182" height="569" alt="Screenshot 2025-07-31 at 8 21 31 PM" src="https://github.com/user-attachments/assets/f08e9039-0e16-4f13-a5f8-4c141eadb69c" />
The final generated images are not very high quality but show good diversity which means the generator has learned on a wide data distribution


## Usage
1. Run the following command to train the model and visualise results
    ```
    python3 main.py
    ```

2. The `results` folder contains images generated for 15 epochs only as the on-device training was done on `mps` which takes up a lot of training time.
3. Go to `config.py` file to change the number of epochs when training.
4. The `config.py` file has support for cuda, mps and cpu.
5. Checkout the detailed [Kaggle Notebook](https://www.kaggle.com/code/saloni0512/dcgan-exp) for this project.
   





## References
* [DCGAN PyTorch Tutorial](https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) 
* [ganhacks repo](https://github.com/soumith/ganhacks)
* [GANs trained by TTUR](https://arxiv.org/pdf/1706.08500)

## Key Takeaways
1. Training GANs takes *days*.
2. Hyperparameter tuning must be done according to the exact usecase and not through random trial and error methods.
3. If discriminator becomes too strong, avoid adding dropouts in its architecture since overfitting might not be the issue here and this will most likely cause misleading results.
4. Label smoothing gives sharper images - recommended for training on low resolution images.
5. Training on 2 GPUs parallely saves a lot of training time!

✨Final note : Had an incredible amount of fun working on this experiment although its based on quite old research, i got a very good taste of computer vision and image generation in depth.

