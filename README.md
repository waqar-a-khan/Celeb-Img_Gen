# **DCGAN Face Generator - Streamlit Web App** 🎨🧑‍🎤

This project is a **web-based application** that uses a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate realistic human faces. The app is built using **Streamlit**, allowing users to generate **16 AI-generated faces in a 4x4 grid** with a single button click.

---

## **📌 Project Overview**

This project aims to **build and deploy a Deep Convolutional GAN (DCGAN)** trained on the **CelebA dataset** to generate synthetic human faces. The final deployment enables users to generate images dynamically through a web-based interface.

### **Dataset: CelebA**
- The **CelebA (CelebFaces Attributes Dataset)** is a large-scale face dataset with over **200,000 celebrity images**.
- It contains diverse facial attributes and variations in pose, lighting, and expressions.
- The dataset was preprocessed and resized to **64x64 pixels** to train the DCGAN efficiently.

🔗 **Dataset Link:** [CelebA on Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

---

## **🚀 Features**

- Loads a **pre-trained DCGAN model** (`checkpoint.pth`) trained on **CelebA dataset**.
- **Streamlit UI** for an easy and interactive experience.
- **Generates 16 faces in a 4x4 grid** with each click.
- **Runs locally** or can be deployed using **Google Colab** with LocalTunnel.
- **Implements a standard DCGAN architecture** with transposed convolutions in the generator and convolutional layers in the discriminator.

---

## **📌 Installation & Running Locally**

### **1️⃣ Install Dependencies**

```bash
pip install streamlit torch torchvision matplotlib
```

### **2️⃣ Clone the Repository**

```bash
git clone https://github.com/yourusername/dcgan-face-generator.git
cd dcgan-face-generator
```

### **3️⃣ Download the Trained Model**

Ensure `checkpoint.pth` is placed in the same directory as `face_gen_web_app.py`. If needed, download it using:

```bash
!wget -O checkpoint.pth https://drive.google.com/file/d/1eXittKO4jnpUzcdd5f5Z99VmKumEjgI7/view?usp=drive_link
```

🔗 **Pre-trained Model URL:** [Download checkpoint.pth](https://drive.google.com/file/d/1eXittKO4jnpUzcdd5f5Z99VmKumEjgI7/view?usp=drive_link)

### **4️⃣ Run the Web App Locally**

```bash
streamlit run face_gen_web_app.py
```

✅ Open [**http://localhost:8501**](http://localhost:8501) in your browser and click **"Generate Image"**!

---

## **🌎 Running in Google Colab**

If you are running this app in **Google Colab**, use the following commands to deploy the Streamlit app with **LocalTunnel**:

```bash
!pip install -q streamlit
!npm install localtunnel
!streamlit run face_gen_web_app.py &>/content/logs.txt & npx localtunnel --port 8501 & curl ipv4.icanhazip.com
```

✅ After running, LocalTunnel will generate a **public link** like:

```
https://yourapp.loca.lt
```

Open this URL in your browser to access the Streamlit app!

---

## **🛠 Technologies Used**

- **Python**
- **PyTorch** (for implementing and training the GAN model)
- **Torchvision** (for image preprocessing and dataset loading)
- **Streamlit** (for the web interface)
- **Matplotlib** (for displaying images)
- **LocalTunnel** (for public deployment in Google Colab)

---

## **💡 Example Output**

After clicking **"Generate Image"**, the app will generate and display **16 AI-generated faces** like this:


---

## **🤝 Contributing**

This project is open for improvements! Feel free to fork this repo, improve it, and submit a pull request. 🚀

---

## **📜 License**

This project is open-source and available under the **MIT License**.

