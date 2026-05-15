import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('logs/jfe/metrics.csv', sep=',')

val_loss = data['val_loss']
val_loss_spkr = data['val_loss_spkr']
val_loss_age = data['val_loss_age']
val_entropy_age = data['val_entropy_age']
val_entropy_spkr = data['val_entropy_spkr']
val_mapc = data['val_mapc']
val_spk_acc = data['val_spk_acc']
val_age_acc = data['val_age_acc']
val_age_leak = data['val_age_leak']
val_id_leak = data['val_id_leak']
epochs = range(1, len(val_loss) + 1)

# draw two pictures, one for losses, one for accuracies
plt.figure(figsize=(10, 8))
plt.plot(epochs, val_loss, label='Val Loss')
plt.plot(epochs, val_loss_spkr, label='Val Loss Spkr')
plt.plot(epochs, val_loss_age, label='Val Loss Age')
plt.plot(epochs, val_entropy_age, label='Val Entropy Age')
plt.plot(epochs, val_entropy_spkr, label='Val Entropy Spkr')
plt.plot(epochs, val_mapc, label='Val MAPC')

plt.title('Validation Losses and Metrics over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss / Metric Value')
plt.legend()
plt.grid()
plt.savefig('validation_losses_metrics.png')
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(epochs, val_spk_acc, label='Val Spk Acc')
plt.plot(epochs, val_age_acc, label='Val Age Acc')
plt.plot(epochs, val_age_leak, label='Val Age Leak')
plt.plot(epochs, val_id_leak, label='Val ID Leak')
plt.title('Validation Accuracies over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid()
plt.savefig('validation_accuracies.png')
plt.show()