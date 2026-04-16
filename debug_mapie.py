import mapie
import mapie.classification as mc

print(f"Mapie version: {mapie.__version__}")
print(f"Contents of mapie.classification: {dir(mc)}")

try:
    from mapie.classification import MapieClassifier
    print("✅ Success: MapieClassifier imported correctly.")
except ImportError as e:
    print(f"❌ Error: {e}")
    # Suggest fix based on version
    if 'MapieClassifier' not in dir(mc):
        print("Heuristic: It seems MapieClassifier is missing from the module.")
