# AI-Answare-Question
Machine Learning AI Web App Using Django:
![Screenshot from 2024-05-13 01-55-10](https://github.com/saeidsaadatigero/AI-Answare-Question/assets/121683582/04f29bc7-8378-4122-8f7c-b8bb496b7078)

این کد یک سیستم پرسش و پاسخ مبتنی بر ماشین لرنینگ را پیاده سازی می‌کند که می‌تواند به کاربر پاسخ‌های مناسب را ارائه دهد. ویژگی‌های اصلی این کد به شرح زیر است:
>1-دریافت دیتاست از فایل CSV و تقسیم آن به مجموعه‌های آموزش و تست
استفاده از تکنیک TF-IDF برای ایجاد بردارهای ویژگی از متن پرسش‌ها
>2-آموزش مدل ماشین یادگیری لجستیک بر روی مجموعه آموزش
>3-ارزیابی عملکرد مدل با استفاده از معیارهایی مانند دقت، فراخوانی و F1-score

>4-ایجاد دو ویو در Django:
-نمایش صفحه اصلی برنامه
-دریافت پرسش از کاربر، پیش‌پردازش آن و پیش‌بینی پاسخ توسط مدل آموزش‌دیده و ارائه پاسخ به کاربر به صورت JSON

First, Import necessary libraries for machine learning:

#### Using `pip`
>
> ```sh
> pip install django
> ```
>
> ```sh
> pip install pandas
> ```
>
> ```sh
> pip install numpy
> ```
>
> ```sh
> pip install scikit-learn
> ```
>
>
> ```sh
> pip install scipy
> ```
>
#### Using `Terminal`
> ```sh
> Python manage.py runserver
> ```
