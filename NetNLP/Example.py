import NeuralNLP as CM
import sys
sys.stdout.encoding

assistant = CM.NLPneural('text.json', model_name="test_model")
assistant.load_model('test_model');
#assistant.train_model(epoh=10, batchSZ=5, log=True)
#assistant.save_model()

done = False
print()
while not done:
    message = input("Enter a message: ")
    if message == "STOP":
        done = True
    else:
        print(assistant.request(message))