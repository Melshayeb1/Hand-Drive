# Hand-Drive
🖐️ Hand Drive Gesture Tool A smart tool that allows you to control games using hand gestures through your webcam. Built with Python and powered by MediaPipe for real-time hand tracking.
🧠 الفكرة العامة للكود:
الكود يتحكم في حركة لوحة المفاتيح (W, A, S, D) بناءً على وضعية وإيماءات اليد الملتقطة بالكاميرا باستخدام مكتبة Mediapipe.
✅ اليد المفتوحة تعني التقدم للأمام (W).
✊ اليد المضمومة تعني التوقف أو الرجوع للخلف (S).
👉 إذا حركت يدك لليمين أو اليسار، تتحكم في الاتجاه (A و D).
مع عرض اليد في نافذة باستخدام matplotlib.

 1. استيراد المكتبات الضرورية
import cv2
import mediapipe as mp
from pynput.keyboard import Controller
import matplotlib.pyplot as plt
import time

cv2: لالتقاط الفيديو من الكاميرا.

mediapipe: لتتبع اليد وتحديد المعالم.

pynput.keyboard: لمحاكاة ضغطات لوحة المفاتيح.

matplotlib.pyplot: لرسم اليد بشكل مرئي.

time: لإضافة تأخير بسيط بين الفريمات.



2. تهيئة Mediapipe والكيبورد

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
keyboard = Controller()


 3. حالة المفاتيح والاتجاه
keys_state = {'w': False, 'a': False, 's': False, 'd': False}
current_direction = None
prev_hand_x = None
يتتبع المفاتيح المضغوطة حتى لا يضغط على نفس الزر أكثر من مرة.

prev_hand_x لتحديد الاتجاه بناءً على حركة اليد أفقياً.


4. دوال التحكم بالمفاتيح
def press_key(key): ...
def release_key(key): ...
def release_all(): ...


press_key: يضغط على زر إذا لم يكن مضغوطًا.

release_key: يحرر الزر إذا كان مضغوطًا.

release_all: يحرر كل المفاتيح عند الحاجة.

✋ 5. رسم اليد

def draw_hand(landmarks): .

تحلل الإيماءات بناءً على:

index_tip, pinky_tip, wrist: لتحديد إذا كانت اليد مفتوحة.

thumb_tip و thumb_ip: للمساعدة في كشف القبضة (fist).

current_x: متوسط موقع الإصبع الصغير والسبابة لتحديد اتجاه الحركة يمين/يسار.

ترجع:

is_open: هل اليد مفتوحة؟

is_fist: هل اليد مضمومة؟

direction: اتجاه الحركة يمين/يسار.

الخطوات:
تفعيل الكاميرا.

تفعيل الرسم التفاعلي (plt.ion()).

قراءة كل فريم.

قلب الصورة أفقيًا (لتحاكي المرآة).

تحويل الصورة لـ RGB لـ Mediapipe.

فحص هل هناك يد في الصورة:

إذا نعم:

ارسم اليد.

حلل الإيماءة والاتجاه.

قرر الإجراء المناسب:

✊ قبضة → اضغط s.

✋ يد مفتوحة → اضغط w واتبع الاتجاه.

إذا لم تتحرك اليد → حافظ على نفس الاتجاه.

إذا لا:

حرر كل الأزرار.

عرض الصورة.

الانتظار قليلاً لتقليل الحمل على المعالج.

إذا ضغط المستخدم على q → إغلاق البرنامج.
