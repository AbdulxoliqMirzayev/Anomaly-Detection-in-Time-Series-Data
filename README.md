# Anomaly-Detection-in-Time-Series-Data

## Loyihaning maqsadi
Ushbu loyiha vaqt qatorlaridagi anomaliyalarni aniqlash uchun **Machine Learning (ML)** usullaridan foydalanadi. Ma'lumotlarni oldindan qayta ishlash, xususiyatlarni yaratish va **Isolation Forest** algoritmi yordamida anomaliyalarni aniqlash jarayonlarini amalga oshiradi.

## Xususiyatlar
- **Ma'lumotlarni oldindan qayta ishlash**: MinMaxScaler yordamida masshtablash.
- **Xususiyatlar yaratish**: Rolling o‘rtacha va standart og‘ish hisoblash.
- **ML modeli**: Isolation Forest algoritmi asosida anomaliyalarni aniqlash.
- **Vizualizatsiya va baholash**: Grafiklarga chiqarish va model natijalarini baholash.

## O‘rnatish
Quyidagi kutubxonalar o‘rnatilganligiga ishonch hosil qiling:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
Yoki Jupyter Notebook'dan foydalaning.

## Foydalanish
1. **CSV fayl yuklash**:
    ```python
    import pandas as pd
    from google.colab import files
    
    print("CSV Faylni yuklang: ")
    uploaded = files.upload()
    
    dataset_name = list(uploaded.keys())[0]
    df = pd.read_csv(dataset_name)
    df.head()
    ```
2. **Ma'lumotlarni masshtablash**:
    ```python
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['value']])
    ```
3. **Isolation Forest yordamida anomaliyalarni aniqlash**:
    ```python
    from sklearn.ensemble import IsolationForest
    
    model = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly'] = model.fit_predict(df[['value']])
    ```
4. **Natijalarni vizualizatsiya qilish**:
    ```python
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10,5))
    plt.plot(df.index, df['value'], label='Vaqt qatori')
    plt.scatter(df.index[df['anomaly'] == -1], df['value'][df['anomaly'] == -1], color='red', label='Anomaliyalar')
    plt.legend()
    plt.show()
    ```

## Qayerda foydalanish mumkin?
- **Moliyaviy ma'lumotlar**: Firibgarlikni aniqlash uchun.
- **Sanoat monitoringi**: Sensor ma'lumotlarida nosozliklarni aniqlash.
- **Tibbiyot**: Biomarkerlar orqali kasalliklarni oldindan aniqlash.
- **Iqlimshunoslik**: Ob-havo o‘zgarishlarini tahlil qilish.



