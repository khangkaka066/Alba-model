"""
Script tạo và tổng hợp dữ liệu huấn luyện đa giai đoạn (Two-Stage Curriculum):
- Giai đoạn 1 (Nền tảng): 45% Edu Text, 35% Code, 10% Math, 10% General Knowledge
- Giai đoạn 2 (Annealing): 40% Toán & Logic, 40% Code sạch, 20% Instruction / Synthetic CoT
"""

import os
import random

DATA_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(DATA_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

PHASE1_FILE = os.path.join(RAW_DIR, "phase1_foundation.txt")
PHASE2_FILE = os.path.join(RAW_DIR, "phase2_annealing.txt")


# -----------------------------------------------------------------------------
# 1. GENERATOR: Edu Text (Văn bản giáo dục & kể chuyện ngữ pháp chuẩn)
# -----------------------------------------------------------------------------
def generate_edu_text_samples(n=1800):
    educational_topics = [
        ("The Water Cycle", "Water on Earth is constantly moving in a cycle. When the sun heats up rivers, lakes, and oceans, liquid water turns into vapor through evaporation. This vapor rises into the atmosphere, cools down, and forms clouds through condensation. When the clouds become heavy with droplets, precipitation falls as rain or snow, returning water to the ground."),
        ("Photosynthesis in Plants", "Plants have the amazing ability to create their own food using sunlight. Inside plant cells, tiny green structures called chloroplasts contain chlorophyll. Chlorophyll absorbs sunlight and combines it with carbon dioxide from the air and water from the soil. This process produces glucose for energy and releases oxygen into the atmosphere for living creatures to breathe."),
        ("Gravity and the Solar System", "Gravity is the invisible force that pulls objects toward one another. The Sun is massive and has a powerful gravitational pull that keeps Earth and all the planets orbiting around it. Similarly, Earth's gravity pulls objects toward its center, which prevents us from floating into space and keeps the atmosphere in place."),
        ("The Human Heart", "The human heart is a powerful muscular organ located in the chest. It acts like a double pump, beating around 100,000 times every day. The right side of the heart pumps oxygen-poor blood to the lungs to pick up fresh oxygen, while the left side pumps oxygen-rich blood throughout the entire body to nourish cells and organs."),
        ("Ecosystems and Biodiversity", "An ecosystem is a community of living organisms interacting with their physical environment. Forest ecosystems, for example, consist of trees, insects, birds, mammals, soil, and water. Every species plays a vital role in maintaining balance. When biodiversity is rich, the ecosystem is resilient and can withstand environmental changes.")
    ]
    
    stories = [
        "Once upon a time, there was a little girl named Lily. She loved exploring the forest near her house. One sunny morning, she saw a little blue bird sitting on a branch. The bird could not fly because its wing was hurt. Lily felt sad and decided to help. She gently carried the bird home and gave it fresh water and bread crumbs. Day after day, she cared for the bird. Soon, the bird healed and sang a joyful song. Lily smiled happily, knowing she had made a wonderful friend.",
        "Tom was a curious boy who loved science and machines. He always wondered how airplanes could fly in the sky. One day, he built a small paper glider using colored paper and a paperclip. He climbed the garden hill and threw it into the gentle breeze. The glider soared high into the air, dancing above the trees. Tom clapped his hands with delight. He promised himself that when he grew up, he would build real planes that fly across the oceans.",
        "In a quiet village, an old clockmaker named Leo lived in a cozy workshop. Every morning, hundreds of clocks chimed together at eight o'clock. One day, a young boy brought a broken pocket watch that belonged to his grandfather. Leo inspected the tiny bronze gears with his magnifying glass. With patient hands, he cleaned the dust, fixed the mainspring, and wound the watch. When the watch ticked steadily again, the boy's eyes lit up with gratitude.",
        "Mia had a puppy named Bruno who loved chasing tennis balls in the park. One afternoon, Bruno accidentally dropped his yellow ball into a small pond. The pond was too deep for Bruno to reach, and he started barking softly. Mia noticed a long tree branch nearby. She carefully fished the ball out and handed it back to Bruno. Bruno wagged his tail excitedly and licked Mia's cheek. It was a lovely afternoon full of play and joy."
    ]

    all_templates = [f"Topic: {title}\nExplanation: {body}" for title, body in educational_topics] + stories
    results = []
    for _ in range(n):
        results.append(random.choice(all_templates))
    return results


# -----------------------------------------------------------------------------
# 2. GENERATOR: Code (Python cơ bản & Code sạch có Docstring)
# -----------------------------------------------------------------------------
def generate_code_samples(n=1400, clean_code=False):
    samples = []
    
    algorithms = [
        ("calculate_factorial", "def factorial(n):\n    if n <= 1:\n        return 1\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result",
         "def factorial(n: int) -> int:\n    \"\"\"Tính giai thừa của một số nguyên không âm n.\"\"\"\n    assert n >= 0, 'n phải là số không âm'\n    if n <= 1:\n        return 1\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result"),
        
        ("find_maximum", "def find_max(numbers):\n    if not numbers:\n        return None\n    max_val = numbers[0]\n    for num in numbers[1:]:\n        if num > max_val:\n            max_val = num\n    return max_val",
         "def find_max(numbers: list[int]) -> int | None:\n    \"\"\"Tìm phần tử có giá trị lớn nhất trong danh sách.\"\"\"\n    if not numbers:\n        return None\n    max_val = numbers[0]\n    for num in numbers[1:]:\n        if num > max_val:\n            max_val = num\n    return max_val"),

        ("check_is_prime", "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
         "def is_prime(n: int) -> bool:\n    \"\"\"Kiểm tra xem số nguyên n có phải là số nguyên tố không.\"\"\"\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"),

        ("reverse_string", "def reverse_text(s):\n    rev = ''\n    for ch in s:\n        rev = ch + rev\n    return rev",
         "def reverse_text(s: str) -> str:\n    \"\"\"Đảo ngược chuỗi ký tự bằng vòng lặp tuyến tính.\"\"\"\n    rev = []\n    for ch in reversed(s):\n        rev.append(ch)\n    return ''.join(rev)"),

        ("sum_even_numbers", "def sum_evens(nums):\n    total = 0\n    for x in nums:\n        if x % 2 == 0:\n            total += x\n    return total",
         "def sum_evens(nums: list[int]) -> int:\n    \"\"\"Tính tổng tất cả các số chẵn trong danh sách.\"\"\"\n    total = 0\n    for x in nums:\n        if x % 2 == 0:\n            total += x\n    return total"),

        ("binary_search", "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1",
         "def binary_search(arr: list[int], target: int) -> int:\n    \"\"\"Tìm kiếm nhị phân phần tử target trong mảng đã sắp xếp.\"\"\"\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1")
    ]
    
    for _ in range(n):
        name, basic, clean = random.choice(algorithms)
        code_str = clean if clean_code else basic
        sample = f"# Python Implementation: {name}\n{code_str}"
        samples.append(sample)
    return samples


# -----------------------------------------------------------------------------
# 3. GENERATOR: Math & Logic (Toán cơ bản & Toán logic chuyên sâu)
# -----------------------------------------------------------------------------
def generate_math_samples(n=400, deep_logic=False):
    samples = []
    for _ in range(n):
        if not deep_logic:
            # Phép tính số học cơ bản
            op = random.choice(["+", "-", "*"])
            if op == "+":
                a, b = random.randint(10, 999), random.randint(10, 999)
                ans = a + b
                text = f"Math Problem: Calculate {a} + {b}.\nCalculation: {a} + {b} = {ans}\nResult: {ans}"
            elif op == "-":
                a = random.randint(50, 999)
                b = random.randint(10, a)
                ans = a - b
                text = f"Math Problem: Calculate {a} - {b}.\nCalculation: {a} - {b} = {ans}\nResult: {ans}"
            else:
                a, b = random.randint(2, 25), random.randint(2, 12)
                ans = a * b
                text = f"Math Problem: Calculate {a} * {b}.\nCalculation: {a} * {b} = {ans}\nResult: {ans}"
        else:
            # Toán logic sâu (hệ phương trình đơn giản, tìm ẩn số, bất đẳng thức)
            p_type = random.choice(["solve_x", "inequality", "ratio"])
            if p_type == "solve_x":
                x_val = random.randint(2, 20)
                m = random.randint(2, 9)
                b = random.randint(1, 30)
                c = m * x_val + b
                text = (
                    f"Logic Problem: Solve for x in the equation: {m} * x + {b} = {c}.\n"
                    f"Derivation: Subtract {b} from both sides: {m} * x = {c - b}.\n"
                    f"Divide both sides by {m}: x = {c - b} / {m} = {x_val}.\n"
                    f"Conclusion: The solution is x = {x_val}."
                )
            elif p_type == "inequality":
                x = random.randint(10, 99)
                y = random.randint(10, 99)
                diff = abs(x - y)
                rel = "greater than" if x > y else ("less than" if x < y else "equal to")
                text = (
                    f"Logic Problem: Given A = {x} and B = {y}, determine the relational order and difference.\n"
                    f"Comparison: A is {rel} B.\n"
                    f"Absolute difference: |A - B| = |{x} - {y}| = {diff}.\n"
                    f"Conclusion: A is {rel} B by {diff} units."
                )
            else:
                # Tỉ lệ số học
                r = random.randint(2, 6)
                base = random.randint(5, 40)
                scaled = base * r
                text = (
                    f"Logic Problem: In a mixture, the ratio of component A to component B is 1:{r}.\n"
                    f"If there are {base} units of A, how many units of B are required?\n"
                    f"Calculation: B = {base} * {r} = {scaled}.\n"
                    f"Conclusion: Exactly {scaled} units of component B are needed."
                )
        samples.append(text)
    return samples


# -----------------------------------------------------------------------------
# 4. GENERATOR: General Knowledge (Tri thức tổng quát)
# -----------------------------------------------------------------------------
def generate_general_knowledge_samples(n=400):
    facts = [
        ("What is the capital city of France?", "The capital of France is Paris, located along the Seine River and famous for landmarks such as the Eiffel Tower."),
        ("Which planet is known as the Red Planet?", "Mars is known as the Red Planet because its surface contains abundant iron oxide (rust), giving it a reddish hue."),
        ("What is the speed of light in vacuum?", "The speed of light in vacuum is approximately 299,792 kilometers per second (about 300,000 km/s)."),
        ("What chemical compound has the formula H2O?", "The chemical formula H2O represents water, consisting of two hydrogen atoms bonded to one oxygen atom."),
        ("Which is the largest ocean on Earth?", "The Pacific Ocean is the largest and deepest ocean on Earth, covering more than 30 percent of the planet's surface."),
        ("How many continents are there on Earth?", "Earth is traditionally divided into seven continents: Asia, Africa, North America, South America, Antarctica, Europe, and Australia."),
        ("What is the process by which liquid turns into gas?", "The process of a liquid transforming into a gas is called vaporization or evaporation when it occurs at the surface."),
        ("Why does the Moon have different phases?", "The Moon appears in different phases because it orbits the Earth, causing different portions of its sunlit side to be visible from our planet.")
    ]
    results = []
    for _ in range(n):
        q, a = random.choice(facts)
        results.append(f"Knowledge Query: {q}\nFactual Answer: {a}")
    return results


# -----------------------------------------------------------------------------
# 5. GENERATOR: Instruction / Synthetic Chain-of-Thought (<think> ... </think>)
# -----------------------------------------------------------------------------
def generate_synthetic_cot_samples(n=400):
    samples = []
    names = ["Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Henry"]
    items = ["apples", "books", "candies", "pens", "stickers", "coins", "marbles", "cookies"]
    
    for _ in range(n):
        name1, name2 = random.sample(names, 2)
        item = random.choice(items)
        n1 = random.randint(10, 60)
        n2 = random.randint(2, 25)
        n3 = random.randint(3, 20)
        total = n1 - n2 + n3
        
        text = (
            f"Instruction: Solve the multi-step word problem with detailed logical chain of thought.\n"
            f"Question: {name1} has {n1} {item}. {name1} gives {n2} {item} to {name2} and buys {n3} more {item}. "
            f"How many {item} does {name1} have now?\n"
            f"<think>\n"
            f"Step 1: Identify initial amount: {name1} starts with {n1} {item}.\n"
            f"Step 2: Calculate quantity after giving {n2} {item} to {name2}: {n1} - {n2} = {n1 - n2}.\n"
            f"Step 3: Calculate final quantity after adding {n3} newly purchased {item}: {n1 - n2} + {n3} = {total}.\n"
            f"</think>\n"
            f"Answer: {name1} now has {total} {item}."
        )
        samples.append(text)
    return samples


# -----------------------------------------------------------------------------
# MAIN: Tổng hợp dữ liệu theo đúng tỷ lệ 2 giai đoạn
# -----------------------------------------------------------------------------
def main():
    random.seed(42)
    print("=" * 65)
    print("BẮT ĐẦU TẠO DỮ LIỆU ĐA GIAI ĐOẠN (TWO-STAGE CURRICULUM)")
    print("=" * 65)

    # -------------------------------------------------------------------------
    # GIAI ĐOẠN 1: NỀN TẢNG (Tổng 4.000 mẫu)
    # Tỷ lệ: 45% Edu Text | 35% Code | 10% Math | 10% General Knowledge
    # -------------------------------------------------------------------------
    p1_edu = generate_edu_text_samples(n=1800)       # 45%
    p1_code = generate_code_samples(n=1400, clean_code=False) # 35%
    p1_math = generate_math_samples(n=400, deep_logic=False)  # 10%
    p1_gk = generate_general_knowledge_samples(n=400)         # 10%

    phase1_data = p1_edu + p1_code + p1_math + p1_gk
    random.shuffle(phase1_data)

    with open(PHASE1_FILE, "w", encoding="utf-8") as f:
        for item in phase1_data:
            f.write(item.strip() + "\n\n")

    print(f"✅ Giai đoạn 1 (Nền tảng): {len(phase1_data):,} mẫu ({os.path.getsize(PHASE1_FILE)/1024:.2f} KB)")
    print(f"   - 45% Edu Text:          {len(p1_edu):,} mẫu")
    print(f"   - 35% Code:              {len(p1_code):,} mẫu")
    print(f"   - 10% Math:              {len(p1_math):,} mẫu")
    print(f"   - 10% General Knowledge: {len(p1_gk):,} mẫu")

    # -------------------------------------------------------------------------
    # GIAI ĐOẠN 2: ANNEALING / TÔI LUYỆN (Tổng 2.000 mẫu)
    # Tỷ lệ: 40% Toán & Logic | 40% Code sạch | 20% Instruction / Synthetic CoT
    # -------------------------------------------------------------------------
    p2_logic = generate_math_samples(n=800, deep_logic=True)   # 40%
    p2_clean_code = generate_code_samples(n=800, clean_code=True) # 40%
    p2_cot = generate_synthetic_cot_samples(n=400)             # 20%

    phase2_data = p2_logic + p2_clean_code + p2_cot
    random.shuffle(phase2_data)

    with open(PHASE2_FILE, "w", encoding="utf-8") as f:
        for item in phase2_data:
            f.write(item.strip() + "\n\n")

    print(f"\n✅ Giai đoạn 2 (Annealing): {len(phase2_data):,} mẫu ({os.path.getsize(PHASE2_FILE)/1024:.2f} KB)")
    print(f"   - 40% Toán & Logic:      {len(p2_logic):,} mẫu")
    print(f"   - 40% Code sạch:         {len(p2_clean_code):,} mẫu")
    print(f"   - 20% Synthetic CoT:     {len(p2_cot):,} mẫu")
    print("=" * 65)


if __name__ == "__main__":
    main()
