from backend import fingerprint

hash_val = fingerprint.fingerprint_image("uploads/test.jpg")
print("Fingerprint:", hash_val)
