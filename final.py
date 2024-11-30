import time
import tracemalloc
from collections import deque
import heapq
import networkx as nx
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import Calendar
from datetime import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import time
import math
from collections import deque
import heapq
import tracemalloc
import random

fixed_positions = {
    "Quận 4": (3, 6),    # Center, dễ thấy các quận khác xung quanh
    "Quận 2": (12, 7),   # Đông Bắc
    "Quận 3": (5, 7),    # Bắc, gần Quận 1
    "Quận 1": (8, 5),    # Trung tâm, phía Nam
    "Quận 5": (10, 4),   # Tây Nam, gần Quận 6
    "Quận 6": (7, 3),    # Tây, gần Bình Tân
    "Quận 7": (11, 3),   # Nam, gần Quận 5 và Bình Chánh
    "Quận 8": (6, 2),    # Tây Nam, gần Quận 7
    "Quận 9": (13, 5),   # Đông, gần Thủ Đức
    "Quận 10": (4, 5),   # Bắc, gần Quận 1
    "Quận 11": (9, 6),   # Trung, gần Quận 5 và Quận 6
    "Quận 12": (12, 4),  # Tây Bắc, gần Gò Vấp
    "Tân Bình": (5, 4),  # Tây, gần Quận 1 và Quận 3
    "Tân Phú": (8, 2),   # Tây, gần Quận 6
    "Bình Tân": (7, 1),  # Tây Nam, gần Quận 8 và Bình Chánh
    "Gò Vấp": (3, 3),    # Bắc, gần Quận 12
    "Bình Thạnh": (4, 3),# Gần Quận 1 và Quận 3
    "Phú Nhuận": (3, 5), # Gần Quận 1, phía Đông
    "Thủ Đức": (14, 6)   # Đông Bắc, gần Quận 2 và Quận 9
}

# List of edges and their attributes
edges = [
    # Các cạnh đã có
    ("Quận 1", "Quận 2", {"weight": 5, "capacity": 100, "fuel_cost": 25000, "distance": 6}),
    ("Quận 2", "Quận 1", {"weight": 5, "capacity": 100, "fuel_cost": 25000, "distance": 6}),
    ("Quận 1", "Quận 3", {"weight": 8, "capacity": 200, "fuel_cost": 35000, "distance": 10}),
    ("Quận 3", "Quận 1", {"weight": 8, "capacity": 200, "fuel_cost": 35000, "distance": 10}),
    ("Quận 1", "Quận 5", {"weight": 10, "capacity": 150, "fuel_cost": 25000, "distance": 12}),
    ("Quận 5", "Quận 1", {"weight": 10, "capacity": 150, "fuel_cost": 25000, "distance": 12}),
    ("Quận 2", "Quận 6", {"weight": 7, "capacity": 120, "fuel_cost": 25000, "distance": 8}),
    ("Quận 6", "Quận 2", {"weight": 7, "capacity": 120, "fuel_cost": 25000, "distance": 8}),
    ("Quận 3", "Quận 4", {"weight": 3, "capacity": 80, "fuel_cost": 25000, "distance": 5}),
    ("Quận 5", "Quận 3", {"weight": 5, "capacity": 80, "fuel_cost": 25000, "distance": 5}),
    ("Quận 8", "Quận 7", {"weight": 6, "capacity": 70, "fuel_cost": 25000, "distance": 7}),
    ("Quận 7", "Quận 8",{"weight": 6, "capacity": 70, "fuel_cost": 25000, "distance": 7}),
    ("Quận 10", "Quận 1", {"weight": 4, "capacity": 90, "fuel_cost": 25000, "distance": 5}),
    ("Quận 1", "Quận 10", {"weight": 4, "capacity": 90, "fuel_cost": 25000, "distance": 5}),
    ("Quận 6", "Quận 11", {"weight": 2, "capacity": 60, "fuel_cost": 25000, "distance": 4}),
    ("Quận 12", "Quận 2", {"weight": 2, "capacity": 60, "fuel_cost": 25000, "distance": 4}),
    ("Tân Phú", "Quận 6", {"weight": 2, "capacity": 50, "fuel_cost": 25000, "distance": 3}),
    ("Quận 1", "Tân Bình",  {"weight": 7, "capacity": 50, "fuel_cost": 25000, "distance": 3}),
    ("Quận 6", "Tân Phú", {"weight": 3, "capacity": 100, "fuel_cost": 25000, "distance": 5}),
    ("Gò Vấp", "Phú Nhuận", {"weight": 6, "capacity": 120, "fuel_cost": 26000, "distance": 9}),
    ("Phú Nhuận", "Gò Vấp", {"weight": 6, "capacity": 120, "fuel_cost": 26000, "distance": 9}),   
    ("Tân Bình", "Bình Thạnh", {"weight": 7, "capacity": 130, "fuel_cost": 27000, "distance": 11}),
    ("Bình Thạnh", "Tân Bình", {"weight": 7, "capacity": 130, "fuel_cost": 27000, "distance": 11}),
    ("Bình Thạnh", "Gò Vấp", {"weight": 6, "capacity": 120, "fuel_cost": 22000, "distance": 8, "traffic_level": "medium"}),
    ("Gò Vấp", "Bình Thạnh", {"weight": 6, "capacity": 120, "fuel_cost": 22000, "distance": 8, "traffic_level": "medium"}),     
    ("Bình Thạnh", "Phú Nhuận", {"weight": 4, "capacity": 90, "fuel_cost": 18000, "distance": 5, "traffic_level": "low"}),
    ("Phú Nhuận", "Bình Thạnh", {"weight": 4, "capacity": 90, "fuel_cost": 18000, "distance": 5, "traffic_level": "low"}),
    ("Tân Bình", "Bình Thạnh", {"weight": 7, "capacity": 110, "fuel_cost": 23000, "distance": 10, "traffic_level": "high"}),
    ("Bình Thạnh", "Tân Bình", {"weight": 7, "capacity": 110, "fuel_cost": 23000, "distance": 10, "traffic_level": "high"}),
    ("Bình Tân", "Phú Nhuận", {"weight": 8, "capacity": 140, "fuel_cost": 26000, "distance": 12, "traffic_level": "high"}),
    ("Phú Nhuận", "Bình Tân", {"weight": 8, "capacity": 140, "fuel_cost": 26000, "distance": 12, "traffic_level": "high"}),
    ("Gò Vấp", "Tân Bình", {"weight": 5, "capacity": 100, "fuel_cost": 24000, "distance": 7, "traffic_level": "medium"}),
    ("Tân Bình", "Gò Vấp", {"weight": 5, "capacity": 100, "fuel_cost": 24000, "distance": 7, "traffic_level": "medium"}),
    ("Bình Tân", "Gò Vấp", {"weight": 9, "capacity": 130, "fuel_cost": 28000, "distance": 13, "traffic_level": "high"}),
    ("Gò Vấp", "Bình Tân", {"weight": 9, "capacity": 130, "fuel_cost": 28000, "distance": 13, "traffic_level": "high"}),
]

# Define Pathfinding Algorithms
class Pathfinding:
    @staticmethod
    def backtracking(graph, source, target, path=None, min_path=None, min_distance=float('inf'), total_distance=0):
        if path is None:
            path = []
        path = path + [source]
        if source == target:
            if total_distance < min_distance:
                return path, total_distance
            return min_path, min_distance
        if source not in graph:
            return min_path, min_distance
        for neighbor in graph[source]:
            if neighbor not in path:
                edge_distance = graph[source][neighbor].get('weight', 0)
                new_distance = total_distance + edge_distance
                candidate_path, candidate_distance = Pathfinding.backtracking(
                    graph, neighbor, target, path, min_path, min_distance, new_distance
                )
                if candidate_distance < min_distance:
                    min_path, min_distance = candidate_path, candidate_distance
        return min_path, min_distance

    @staticmethod
    def backtracking_dfs(graph, source, target, visited=None):
        if visited is None:
            visited = set()
        visited.add(source)
        if source == target:
            return [source]
        for neighbor in graph.get(source, []):
            if neighbor not in visited:
                path = Pathfinding.backtracking_dfs(graph, neighbor, target, visited)
                if path:
                    return [source] + path
        visited.remove(source)
        return None

    @staticmethod
    def backtracking_bfs(graph, start, target):
        queue = deque([(start, [start])])
        visited = set()
        while queue:
            current, path = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            if current == target:
                return path
            for neighbor in graph[current]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
        return None

    @staticmethod
    def a_star(graph, start, target, heuristic):
        open_set = [(0, start, [start])]
        g_costs = {start: 0}
        while open_set:
            _, current, path = heapq.heappop(open_set)
            if current == target:
                return path, g_costs[current]
            for neighbor, attr in graph[current].items():
                new_cost = g_costs[current] + attr.get('weight', 1)
                if neighbor not in g_costs or new_cost < g_costs[neighbor]:
                    g_costs[neighbor] = new_cost
                    f_cost = new_cost + heuristic(neighbor, target)
                    heapq.heappush(open_set, (f_cost, neighbor, path + [neighbor]))
        return None, float('inf')

    @staticmethod
    def dijkstra(graph, start, target):
        pq = [(0, start, [])]
        visited = set()
        while pq:
            (cost, current, path) = heapq.heappop(pq)
            if current in visited:
                continue
            path = path + [current]
            visited.add(current)
            if current == target:
                return path, cost
            for neighbor, attr in graph[current].items():
                if neighbor not in visited:
                    weight = attr.get('weight', 1)
                    heapq.heappush(pq, (cost + weight, neighbor, path))
        return None, float('inf')

# Convert edges to graph
def convert_edges_to_graph(edges):
    graph = {}
    for u, v, attr in edges:
        if u not in graph:
            graph[u] = {}
        graph[u][v] = {
            'weight': attr.get('weight', 1),
            'capacity': attr.get('capacity', 0),
            'fuel_cost': attr.get('fuel_cost', 0),
            'distance': attr.get('distance', 1),
        }
        if v not in graph:
            graph[v] = {}
        graph[v][u] = {
            'weight': attr.get('weight', 1),
            'capacity': attr.get('capacity', 0),
            'fuel_cost': attr.get('fuel_cost', 0),
            'distance': attr.get('distance', 1),
        }
    return graph



# Heuristic function for A* (Manhattan Distance)
def heuristic(node, target):
    heuristic_values = {'Quận 1': 7, 'Quận 2': 6, 'Quận 3': 2, 'Quận 4': 1, 'Quận 5': 0}
    return heuristic_values.get(node, 0)

# Test algorithms
def test_algorithms(graph, source, target):
    results = {}

    # Backtracking
    start_time = time.time()
    backtracking_path, backtracking_cost = Pathfinding.backtracking(graph, source, target)
    end_time = time.time()
    results['Backtracking'] = {
        'Path': backtracking_path,
        'Cost': backtracking_cost,
        'Time': end_time - start_time
    }

    # DFS
    start_time = time.time()
    dfs_path = Pathfinding.backtracking_dfs(graph, source, target)
    end_time = time.time()
    results['DFS'] = {
        'Path': dfs_path,
        'Time': end_time - start_time
    }

    # BFS
    start_time = time.time()
    bfs_path = Pathfinding.backtracking_bfs(graph, source, target)
    end_time = time.time()
    results['BFS'] = {
        'Path': bfs_path,
        'Time': end_time - start_time
    }

    # Dijkstra
    start_time = time.time()
    dijkstra_path, dijkstra_cost = Pathfinding.dijkstra(graph, source, target)
    end_time = time.time()
    results['Dijkstra'] = {
        'Path': dijkstra_path,
        'Cost': dijkstra_cost,
        'Time': end_time - start_time
    }

    # A* Algorithm
    tracemalloc.start()
    start_time = time.time()
    a_star_path, a_star_cost = Pathfinding.a_star(graph, source, target, heuristic)
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results['A*'] = {
        'Path': a_star_path,
        'Cost': a_star_cost,
        'Time': end_time - start_time,
        'Memory Usage': f"Current={current / 10**6:.3f}MB, Peak={peak / 10**6:.3f}MB"
    }

    return results

# Cập nhật TrafficGraph để hỗ trợ allowed_vehicles
class TrafficGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        for u, v, attr in edges:
            attr.setdefault("allowed_vehicles", ["Car", "Bike", "Truck"])  # Mặc định cho phép tất cả phương tiện
            self.graph.add_edge(u, v, **attr)
            self.graph[u][v]['flow'] = 0  # Khởi tạo lưu lượng giao thông là 0


   
    def calculate_total_cost(self, path, vehicle_type):
        total_fuel_cost = 0
        total_distance = 0
        fuel_price = 25000  # Giá nhiên liệu (VND/lít)
        
        vehicle_efficiency = {
            "Xe hơi": {"factor": 1.0, "fuel_consumption": 0.1},   # 10 km/lít
            "Xe máy": {"factor": 0.5, "fuel_consumption": 0.25},  # 4 km/lít
            "Xe buýt": {"factor": 1.5, "fuel_consumption": 0.05}  # 20 km/lít
        }

        vehicle_data = vehicle_efficiency.get(vehicle_type, {"factor": 1.0, "fuel_consumption": 0.1})
        
        for u, v in zip(path[:-1], path[1:]):
            try:
                if u not in self.graph or v not in self.graph[u]:
                    raise KeyError(f"Cạnh từ '{u}' đến '{v}' không tồn tại trong đồ thị.")
                edge_data = self.graph[u][v]
                
                distance = edge_data.get('distance', 10)  # Mặc định 10km nếu không có
                fuel_consumed = distance * vehicle_data['fuel_consumption']
                edge_fuel_cost = fuel_consumed * fuel_price * vehicle_data['factor']
                
                total_fuel_cost += edge_fuel_cost
                total_distance += distance

            except KeyError as e:
                print(f"Lỗi: {e}")
                return {"total_fuel_cost": float('inf'), "total_distance": float('inf')}

        return {
            "total_fuel_cost": round(total_fuel_cost, 2),
            "total_distance": round(total_distance, 2)
        }
    
    def update_traffic(self, multiplier):
        for u, v, data in self.graph.edges(data=True):
            data['weight'] *= multiplier

    def set_edge_flow(self, u, v, flow):
        """Update the flow for a specific edge."""
        if u in self.graph and v in self.graph[u]:
            self.graph[u][v]['flow'] = flow
        else:
            raise ValueError(f"Edge ({u}, {v}) does not exist in the graph.")        

    def get_edge_weight(self, u, v, vehicle_type, current_time, is_holiday):
        if u not in self.graph or v not in self.graph[u]:
            raise ValueError(f"Edge ({u}, {v}) does not exist in the graph.")
        
        # Kiểm tra phương tiện có được phép không
        allowed_vehicles = self.graph[u][v].get("allowed_vehicles", [])
        if vehicle_type not in allowed_vehicles:
            return float('inf')  # Không thể đi qua, coi như vô hạn thời gian

        adjustment = 0
        if vehicle_type == "Car":
            adjustment += 0
        elif vehicle_type == "Bike":
            adjustment -= 2
        elif vehicle_type == "Truck":
            adjustment += 3
        else:
            raise ValueError(f"Unknown vehicle type: {vehicle_type}")

        if is_holiday:
            adjustment += 3

        if 7 <= current_time <= 9 or 17 <= current_time <= 19:  # Giờ cao điểm
            adjustment += 5

        edge_weight = self.graph[u][v]['weight'] + adjustment
        edge_weight = max(edge_weight, 0)  # Tránh giá trị âm

                # Tính toán thời gian trên cạnh
        edge_weight = self.graph[u][v]['weight'] + adjustment

        # Đảm bảo thời gian không âm
        edge_weight = max(0, edge_weight)

        # Làm tròn lên thời gian
        return math.ceil(edge_weight)           
    
    def update_weather_effect(self, weather):
        """Cập nhật trọng số đồ thị dựa trên thời tiết và làm tròn lên."""
        weather_multiplier = {
            "Nắng": 1.0,
            "Mưa vừa": 1.2,
            "Mưa to": 1.5
        }
        multiplier = weather_multiplier.get(weather, 1.0)
        for u, v, data in self.graph.edges(data=True):
            data['weight'] = math.ceil(data['weight'] * multiplier)

# Thêm mô phỏng tự động trong TrafficSimulationApp
class TrafficSimulationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Simulation")
        self.traffic_graph = TrafficGraph()

        self.create_widgets()

        self.current_time = 0  # Thời gian hiện tại trong mô phỏng
        self.edge_objects = []

        # Khởi tạo figure và axis cho matplotlib
        self.fig, self.ax = plt.subplots(figsize=(5, 4))  # Tạo cửa sổ đồ thị duy nhất
        plt.ion()  # Bật chế độ interactive để vẽ lại đồ thị mà không mở thêm cửa sổ


        self.is_simulating = False
        
    def create_widgets(self):

        frame = ttk.Frame(self.root)
        frame.pack(padx=200, pady=50, fill=tk.BOTH, expand=True)

        # Giả sử bạn có một danh sách các quận trong đồ thị để điền vào Combobox
        districts = ["Quận 1", "Quận 2", "Quận 3", "Quận 4", "Quận 5", "Quận 6", "Quận 7", "Quận 8", "Quận 9", "Quận 10", "Quận 11", "Quận 12", "Tân Bình", "Tân Phú", "Bình Tân", "Gò Vấp", "Bình Thạnh", "Phú Nhuận", "Thủ Đức"]

        # Sử dụng Combobox thay vì Entry
        ttk.Label(frame, text="Start Point:").grid(row=0, column=0, sticky=tk.W)
        self.start_entry = ttk.Combobox(frame, values=districts)
        self.start_entry.grid(row=0, column=1, pady=5)

        ttk.Label(frame, text="End Point:").grid(row=1, column=0, sticky=tk.W)
        self.end_entry = ttk.Combobox(frame, values=districts)
        self.end_entry.grid(row=1, column=1, pady=5)

        ttk.Label(frame, text="Vehicle Type:").grid(row=2, column=0, sticky=tk.W)
        self.vehicle_combo = ttk.Combobox(frame, values=["Car", "Bike", "Truck"])
        self.vehicle_combo.grid(row=2, column=1, pady=5)

        ttk.Label(frame, text="Time:").grid(row=3, column=0, sticky=tk.W)
        self.time_combo = ttk.Combobox(frame, values=[str(i).zfill(2) for i in range(24)], state="readonly")
        self.time_combo.grid(row=3, column=1, pady=5)

        ttk.Label(frame, text="Thời tiết:").grid(row=4, column=0, sticky=tk.W)
        self.weather_combo = ttk.Combobox(frame, values=["Nắng", "Mưa vừa", "Mưa to"], state="readonly")
        self.weather_combo.grid(row=4, column=1, pady=5)

        self.calendar = Calendar(frame)
        self.calendar.grid(row=5, column=0, columnspan=2, pady=10)

        self.simulate_button = ttk.Button(frame, text="Simulate", command=self.simulate)
        self.simulate_button.grid(row=6, column=0, columnspan=2, pady=5)

        self.stop_button = ttk.Button(frame, text="Stop Simulation", command=self.stop_simulation)
        self.stop_button.grid(row=7, column=0, columnspan=2, pady=5)

        # Nút chạy mô phỏng tự động
        self.auto_simulate_button = ttk.Button(frame, text="Start Auto Simulation", command=self.start_auto_simulation)
        self.auto_simulate_button.grid(row=10, column=0, columnspan=2, pady=5)

        #Nút so sánh
        self.auto_simulate_button = ttk.Button(frame, text="Start So sánh", command=self.run_pathfinding)
        self.auto_simulate_button.grid(row=11, column=0, columnspan=2, pady=5)

        #Nút hiệ thị đồ thị ban đầu
        self.initial_graph_button = ttk.Button(frame, text="Bản đồ", command=self.visualize_graph)
        self.initial_graph_button.grid(row=12, column=0, columnspan=2, pady=5)

        self.time_slider = ttk.Scale(frame, from_=0, to=23, orient=tk.HORIZONTAL, command=self.update_time)
        self.time_slider.grid(row=13, column=0, columnspan=2, pady=10)
        self.time_label = ttk.Label(frame, text="Time: 0")
        self.time_label.grid(row=14, column=0, columnspan=2, pady=5)
    
    def create_comparison_tabs(self, paths):
        self.path_notebook = ttk.Notebook(self.root)
        self.path_notebook.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        vehicle_type = self.vehicle_combo.get()
        time_str = self.time_combo.get()
        current_time = int(time_str)
        is_holiday = self.is_holiday()

        # Tính toán chi phí và thời gian cho tất cả các đường đi
        path_details = []
        for path in paths:
            cost_details = self.traffic_graph.calculate_total_cost(path, vehicle_type)
            total_time = self.calculate_total_time(path, vehicle_type, current_time, is_holiday)
            path_details.append({
                "path": path,
                "total_distance": cost_details['total_distance'],
                "total_fuel_cost": cost_details['total_fuel_cost'],
                "total_time": total_time
            })

        # Xác định tuyến đường ngắn nhất (tùy thuộc vào tiêu chí, ví dụ: quãng đường)
        shortest_path = min(path_details, key=lambda x: x['total_distance'])

        for i, details in enumerate(path_details, 1):
            path_frame = ttk.Frame(self.path_notebook)

            details_frame = ttk.LabelFrame(path_frame, text=f"Đường đi {i}")
            details_frame.pack(padx=10, pady=10, fill=tk.X)

            ttk.Label(details_frame, text=f"Tuyến đường: {' -> '.join(details['path'])}").pack(anchor='w')
            ttk.Label(details_frame, text=f"Thời gian dự kiến: {details['total_time']} phút").pack(anchor='w')
            ttk.Label(details_frame, text=f"Tổng quãng đường: {details['total_distance']} km").pack(anchor='w')
            ttk.Label(details_frame, text=f"Chi phí nhiên liệu: {details['total_fuel_cost']} VND").pack(anchor='w')

            # Nút hiển thị bản đồ
            visualize_button = ttk.Button(details_frame, 
                                        text="Hiển thị bản đồ", 
                                        command=lambda p=details['path']: self.visualize_graph(p))
            visualize_button.pack(anchor='w', pady=5)

            # Đánh dấu đường ngắn nhất
            if details == shortest_path:
                self.path_notebook.add(path_frame, text=f"Đường {i} (Ngắn nhất)")
            else:
                self.path_notebook.add(path_frame, text=f"Đường {i}")

    def simulate(self):
        start = self.start_entry.get()
        end = self.end_entry.get()        
        vehicle_type = self.vehicle_combo.get()
        time_str = self.time_combo.get()
        weather = self.weather_combo.get()

        if not start or not end or not vehicle_type or not time_str:
            messagebox.showerror("Lỗi nhập dữ liệu", "Vui lòng điền đầy đủ các dữ liệu.")
            return

        try:
            current_time = int(time_str)
        except ValueError:
            messagebox.showerror("Lỗi nhập dữ liệu", "Định dạng thời gian không hợp lệ.")
            return

        if current_time < 0 or current_time > 23:
            messagebox.showerror("Lỗi nhập dữ liệu", "Thời gian phải từ 0 đến 23.")
            return

        # Cập nhật ảnh hưởng thời tiết
        self.traffic_graph.update_weather_effect(weather)

        # Destroy existing notebook if it exists
        if hasattr(self, 'path_notebook'):
            self.path_notebook.destroy()

        is_holiday = self.is_holiday()

        # Tìm đường tối ưu bằng backtracking
        path, total_distance = Pathfinding.backtracking(self.traffic_graph.graph, start, end)

        if path:
            cost_details = self.traffic_graph.calculate_total_cost(path, vehicle_type)
            total_time = self.calculate_total_time(path, vehicle_type, current_time, is_holiday)

            result_message = (
                f"Thời tiết: {weather}\n"
                f"Đường đi: {' -> '.join(path)}\n"
                f"Thời gian dự kiến: {math.ceil(total_time)} phút\n"  # Làm tròn lên
                f"Tổng quãng đường: {math.ceil(total_distance)} km\n"  # Làm tròn lên
                f"Chi phí nhiên liệu: {math.ceil(cost_details['total_fuel_cost'])} VND\n"  # Làm tròn lên
            )
            messagebox.showinfo("Chi tiết đường đi", result_message)
            self.visualize_graph(path)
        else:
            messagebox.showwarning("Không tìm thấy đường", "Không có đường đi phù hợp.")

        # Tìm nhiều đường đi thay thế
        paths = self.find_alternative_paths(self.start_entry.get(), self.end_entry.get())
        
        if paths:
            self.create_comparison_tabs(paths)
        else:
            messagebox.showwarning("Không tìm thấy đường", "Không có đường đi phù hợp.")        

    def calculate_total_time(self, path, vehicle_type, current_time, is_holiday):
        total_time = 0
        for u, v in zip(path[:-1], path[1:]):
            try:
                edge_weight = self.traffic_graph.get_edge_weight(u, v, vehicle_type, current_time, is_holiday)
                total_time += edge_weight
            except KeyError as e:
                print(f"Lỗi: {e}")
                return float('inf')  # Nếu có lỗi, trả về giá trị vô hạn
        return total_time

    def is_holiday(self):
        today = datetime.today().date()
        holidays = [(1, 1), (4, 30), (5, 1), (9, 2), (12, 31)]
        return (today.month, today.day) in holidays

    def visualize_graph(self, path=None):  # path có thể là None nếu không cần thiết
        plt.figure(figsize=(12, 10))
        pos = fixed_positions  # Hoặc sử dụng layout khác

        edges = self.traffic_graph.graph.edges(data=True)

        edge_colors = []
        for u, v, data in edges:
            if data['flow'] > 0 or (7 <= self.current_time <= 9 or 17 <= self.current_time <= 19):
                edge_colors.append('red')  # Màu đỏ cho giờ cao điểm
            else:
                edge_colors.append('green')  # Màu xanh cho giờ bình thường

        edge_labels = {}
        for u, v, data in self.traffic_graph.graph.edges(data=True):
            edge_labels[(u, v)] = f"Wt: {data['weight']}p\nCap: {data['capacity']}"        

        # Vẽ đồ thị
        nx.draw(self.traffic_graph.graph, pos, with_labels=True, node_color='lightblue', edge_color=edge_colors, width=1.5)

        if path:
            # Nếu có path, vẽ đường đi
            path_edges = list(zip(path, path[1:]))
            path_colors = ['red'] * len(path_edges)  # Vẽ đường đi với màu đỏ
            nx.draw_networkx_edges(self.traffic_graph.graph, pos, edgelist=path_edges, edge_color=path_colors, width=2)

        plt.title(f"Traffic at Time: {self.current_time}")
        plt.draw()
        plt.pause(0.01)

    def update_time(self, event=None):
        self.current_time = int(self.time_slider.get())
        self.time_label.config(text=f"Time: {self.current_time}")
        

        # Nếu cần vẽ lại đồ thị có liên quan đến path, bạn phải tính toán path ở đây
        # Ví dụ:
        start = self.start_entry.get()
        end = self.end_entry.get()
        path = Pathfinding.backtracking(self.traffic_graph.graph, start, end)

        # Vẽ lại đồ thị với hoặc không có path
        self.visualize_graph(path)

##--------------------------------------------------------------------------------------
## -- Tìm tất cả đường đi bằng backtracking -------------------------

    def find_alternative_paths(self, start, end, max_paths=3):
    # Sử dụng thuật toán tìm nhiều đường đi
        alternative_paths = []
        graph = self.traffic_graph.graph.to_undirected()
        
        # Sử dụng networkx để tìm các đường đi khác nhau
        try:
            for path in nx.all_simple_paths(graph, source=start, target=end):
                if path not in alternative_paths:
                    alternative_paths.append(path)
                    if len(alternative_paths) >= max_paths:
                        break
        except nx.NetworkXNoPath:
            return []
        
        return alternative_paths   
    
##--------------------------------------------------------------------------------------
## -- MÔ PHỎNG TỰ ĐỘNG-------------------------    
    def start_auto_simulation(self):
        self.is_simulating = True
        self.auto_simulate()

    def visualize_initial_graph(self):      
        plt.figure(figsize=(5, 3))
        pos = fixed_positions

        # Vẽ tất cả các nút màu lightblue
        node_colors = ['lightblue' for _ in self.traffic_graph.graph.nodes()]

        # Vẽ đồ thị với các nút lightblue và mũi tên màu đen
        nx.draw(self.traffic_graph.graph, pos, 
                with_labels=True, 
                node_color=node_colors, 
                font_weight="bold", 
                font_size=10,
                edge_color='black', 
                width=1, 
                node_size=2000, 
                font_color='black',
                arrows=True,
                arrowsize=20)

        # Tạo nhãn cho các cạnh để hiển thị trọng số và sức chứa
        edge_labels = {}
        for u, v, data in self.traffic_graph.graph.edges(data=True):
            edge_labels[(u, v)] = f"Wt: {data['weight']}p\nCap: {data['capacity']}"

        # Vẽ nhãn cạnh (trọng số và sức chứa)
        nx.draw_networkx_edge_labels(self.traffic_graph.graph, pos, edge_labels=edge_labels)

        plt.title("Initial Traffic Network")
        plt.show()

    def simulate_traffic_congestion(self):
    # Danh sách các cạnh bị ùn tắc
        congested_edges = []

        # Kiểm tra nếu là giờ bình thường (từ 9h đến 17h)
        if 9 <= self.current_time < 17:
            for u, v, data in self.traffic_graph.edges(data=True):
                # Giả sử có 20% cơ hội để một cạnh bị ùn tắc trong giờ bình thường
                if random.random() < 0.2:
                    congested_edges.append((u, v))

                    # Tăng trọng số và giảm công suất để thể hiện sự ùn tắc
                    data['weight'] *= 1.5  # Tăng trọng số, làm cho việc di chuyển trở nên tốn thời gian hơn
                    data['capacity'] = max(0, data['capacity'] - 20)  # Giảm công suất (giới hạn tránh giá trị âm)
                    
        return congested_edges

    def auto_simulate(self):
        if not self.is_simulating:  # Kiểm tra nếu mô phỏng không đang chạy
            return

        # Mỗi giây tương ứng với 1 giờ trong mô phỏng
        self.current_time = (self.current_time + 1) % 24  # Cập nhật thời gian mỗi giờ
        self.time_label.config(text=f"Time: {self.current_time} hours")  # Cập nhật nhãn hiển thị thời gian

        # Xác định các cạnh bị ùn tắc trong giờ bình thường
        congested_edges = self.simulate_traffic_congestion()

        # Cập nhật thanh trượt và nhãn thời gian
        self.time_slider.set(self.current_time)  # Cập nhật thanh trượt
        self.time_label.config(text=f"Time: {self.current_time}")  # Cập nhật nhãn hiển thị thời gian

        # Vẽ lại đồ thị với màu sắc cập nhật
        self.ax.clear()  # Xóa đồ thị cũ trong ax
        pos = nx.spring_layout(self.traffic_graph)  # Tạo layout cho các nút

        # Xác định màu sắc các cạnh dựa trên tình trạng tắc nghẽn
        edge_colors = []
        for u, v, data in self.traffic_graph.edges(data=True):
            if (u, v) in congested_edges:
                edge_colors.append('red')  # Màu đỏ cho các cạnh bị ùn tắc
                self.show_congestion_alert(u, v)
            else:
                edge_colors.append('green')  # Màu xanh cho các cạnh không bị ùn tắc

        # Vẽ đồ thị
        nx.draw(self.traffic_graph, pos, with_labels=True, node_color='lightblue', ax=self.ax, font_weight="bold", edge_color=edge_colors, width=2)

        # Cập nhật nhãn cho các cạnh
        edge_labels = {}
        for u, v, data in self.traffic_graph.edges(data=True):
            edge_labels[(u, v)] = f"Wt: {data['weight']}p\nCap: {data['capacity']}"
        nx.draw_networkx_edge_labels(self.traffic_graph, pos, edge_labels=edge_labels, ax=self.ax)

        # Cập nhật hình ảnh đồ thị sau 1 giây
        self.root.after(1000, self.auto_simulate)  # Gọi lại sau 1 giây để tiếp tục mô phỏng

    def show_congestion_alert(self, u, v):
        # Lựa chọn nguyên nhân ùn tắc ngẫu nhiên
        reasons = ["Tai nạn", "Sự cố giao thông", "Công trình đang thi công", "Đường tắc do mưa"]
        reason = random.choice(reasons)  # Chọn nguyên nhân ngẫu nhiên

        # Hiển thị thông báo về nguyên nhân ùn tắc
        messagebox.showinfo("Ùn tắc giao thông", f"Ùn tắc trên tuyến đường: {u} -> {v}\nNguyên nhân: {reason}")

    def run_simulation(self):
        """
        Chạy mô phỏng tự động cho 24 giờ.
        """
        self.is_simulating = True
        self.auto_simulate()  # Chạy mô phỏng tự động, sẽ tiếp tục qua các giờ
        
    def stop_simulation(self):
        self.is_simulating = False
        messagebox.showinfo("Simulation Stopped", "The simulation has been stopped.")

##-------------------------------------------------------------------------------
#----------  GUI Functionality - So sánh thuật toán ---------------

    def run_pathfinding(self):
        graph = convert_edges_to_graph(edges)

        source = self.start_entry.get()
        target = self.end_entry.get()

        # Kiểm tra tính hợp lệ của source và target
        if source not in graph or target not in graph:
            messagebox.showerror("Lỗi", f"Source ({source}) hoặc Target ({target}) không tồn tại trong đồ thị!")
            return

        # Chạy so sánh thuật toán
        try:
            results = test_algorithms(graph, source, target)

            # Chuẩn bị chuỗi kết quả để hiển thị
            output = ""
            for algo, data in results.items():
                output += f"{algo}:\n"
                output += f"  Path: {data['Path']}\n"
                output += f"  Cost: {data.get('Cost', 'N/A')}\n"
                output += f"  Time: {data['Time']:.12f}s\n"
                if 'Memory Usage' in data:
                    output += f"  Memory Usage: {data['Memory Usage']}\n"
                output += "\n"  # Dòng trống giữa các thuật toán

            # Hiển thị kết quả trên cửa sổ GUI
            messagebox.showinfo("Kết quả thuật toán", output)

        except Exception as e:
            messagebox.showerror("Lỗi thực thi", f"Đã xảy ra lỗi trong khi chạy thuật toán: {e}")

  
if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSimulationApp(root)
    root.mainloop()
