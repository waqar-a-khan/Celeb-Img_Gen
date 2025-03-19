# **DCGAN Face Generator - Streamlit Web App** 🎨🧑‍🎤  
This project is a **web-based application** that uses a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate realistic human faces. The app is built using **Streamlit**, allowing users to generate **16 AI-generated faces in a 4x4 grid** with a single button click.

---

## **🚀 Features**
- Loads a **pre-trained DCGAN model** (`checkpoint.pth`) to generate realistic faces.
- **Streamlit UI** for an easy and interactive experience.
- **Generates 16 faces in a 4x4 grid** with each click.
- **Runs locally** or can be deployed using **Google Colab** with LocalTunnel.

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
Ensure `checkpoint.pth` is placed in the same directory as `gan_web_deployment.py`. If needed, download it using:
```bash
!wget -O checkpoint.pth YOUR_MODEL_URL
```
🔗 **Pre-trained Model URL:** [Download checkpoint.pth]([(https://drive.google.com/file/d/1eXittKO4jnpUzcdd5f5Z99VmKumEjgI7/view?usp=drive_link))]

### **4️⃣ Run the Web App Locally**
```bash
streamlit run gan_web_deployment.py
```
✅ Open **http://localhost:8501** in your browser and click **"Generate Image"**!

---

## **🌎 Running in Google Colab**
If you are running this app in **Google Colab**, use the following commands to deploy the Streamlit app with **LocalTunnel**:

```bash
!pip install -q streamlit
!npm install localtunnel
!streamlit run gan_web_deployment.py &>/content/logs.txt & npx localtunnel --port 8501 & curl ipv4.icanhazip.com
```

✅ After running, LocalTunnel will generate a **public link** like:
```
https://yourapp.loca.lt
```
Open this URL in your browser to access the Streamlit app!

---

## **🛠 Technologies Used**
- **Python**
- **PyTorch** (for loading the trained GAN model)
- **Torchvision** (for image utilities)
- **Streamlit** (for the web interface)
- **Matplotlib** (for displaying images)
- **LocalTunnel** (for public deployment in Google Colab)

---

## **💡 Example Output**
After clicking **"Generate Image"**, the app will generate and display **16 AI-generated faces** like this:

![Example Output](https://your-image-url.com/example.png)

---

## **🤝 Contributing**
Feel free to fork this repo, improve it, and submit a pull request! 🚀

---

## **📜 License**
This project is open-source and available under the **MIT License**.

