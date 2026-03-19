# 1. Import at top
from data.api_client import APIClient

# 2. In PillWheelApp.__init__(), add:
self.api = APIClient()

# 3. Replace _identify_thread() — instead of local FR, send to server:
def _identify_thread(self, frame):
    if not self.api.is_online():
        self.root.after(0, lambda: self._show_error(
            "No connection to server.\nPlease call for assistance."
        ))
        return
    
    data = self.api.identify_patient(frame)
    if data:
        self.root.after(0, lambda: self._on_identified(data))
    else:
        self.root.after(0, lambda: self._on_identity_failed("Face not recognised"))

# 4. In _on_identified(), server response maps directly:
def _on_identified(self, data: dict):
    self.current_patient = data
    # data["prescriptions"][0]["container"] → servo channel
    # data["prescriptions"][0]["quantity"]  → how many pills
    # data["prescriptions"][0]["instructions"] → shown after dispense

# 5. After dispense completes in _dispense_thread(), add:
for rx in patient["prescriptions"]:
    self.api.log_intake(patient["patientId"], rx["medicineId"], rx["quantity"])
    self.api.reduce_stock(rx["medicineName"], rx["quantity"])

if audit_frame is not None:
    self.api.upload_audit_image(patient["patientId"], audit_frame, timestamp)