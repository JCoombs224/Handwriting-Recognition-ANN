updates = []

for i in range(784):
    updates.append(f"a{i} = a{i} / 255.0")

print(f"{','.join(updates)}")