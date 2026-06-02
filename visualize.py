import pygame
import sys
import random
from delivery_env import DeliveryEnvironment

class DeliveryVisualizer:
    def __init__(self, env):
        pygame.init()
        
        self.env = env
        self.grid_size = env.graph.grid_size
        
        # Window setup
        self.margin = 100
        self.cell_size = 120
        self.width = self.margin * 2 + (self.grid_size - 1) * self.cell_size
        self.height = self.margin * 2 + (self.grid_size - 1) * self.cell_size + 150
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Delivery Optimization Simulation')
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (180, 180, 180)
        self.LIGHT_GRAY = (230, 230, 230)
        self.DARK_GRAY = (100, 100, 100)
        
        # Driver colors
        self.DRIVER_COLORS = [
            (220, 20, 60),   # Crimson Red
            (30, 144, 255),  # Dodger Blue
            (50, 205, 50)    # Lime Green
        ]
        
        self.ORANGE = (255, 140, 0)
        self.GOLD = (255, 215, 0)
        
        # Fonts
        self.font_large = pygame.font.Font(None, 28)
        self.font_medium = pygame.font.Font(None, 22)
        self.font_small = pygame.font.Font(None, 18)
        
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        self.speed = 5  # Steps per second
    
    def node_to_pos(self, node):
        """Convert node ID to screen position"""
        row = node // self.grid_size
        col = node % self.grid_size
        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        return x, y
    
    def draw_grid(self):
        """Draw the city grid"""
        # Draw edges first (so they're behind nodes)
        for (n1, n2) in self.env.graph.edges.keys():
            x1, y1 = self.node_to_pos(n1)
            x2, y2 = self.node_to_pos(n2)
            pygame.draw.line(self.screen, self.LIGHT_GRAY, (x1, y1), (x2, y2), 3)
        
        # Draw nodes
        for node in range(self.env.graph.num_nodes):
            x, y = self.node_to_pos(node)
            
            # Node circle
            pygame.draw.circle(self.screen, self.WHITE, (x, y), 20)
            pygame.draw.circle(self.screen, self.GRAY, (x, y), 20, 2)
            
            # Node number
            text = self.font_small.render(str(node), True, self.DARK_GRAY)
            text_rect = text.get_rect(center=(x, y))
            self.screen.blit(text, text_rect)
    
    def draw_drivers(self):
        """Draw drivers at their current positions"""
        for driver in self.env.drivers:
            x, y = self.node_to_pos(driver.current_node)
            
            # Driver circle
            color = self.DRIVER_COLORS[driver.id]
            pygame.draw.circle(self.screen, color, (x, y), 25)
            pygame.draw.circle(self.screen, self.BLACK, (x, y), 25, 2)
            
            # Driver ID
            text = self.font_medium.render(str(driver.id), True, self.WHITE)
            text_rect = text.get_rect(center=(x, y))
            self.screen.blit(text, text_rect)
            
            # Queue indicator (small badge)
            if driver.order_queue:
                badge_x = x + 20
                badge_y = y - 20
                pygame.draw.circle(self.screen, self.GOLD, (badge_x, badge_y), 12)
                pygame.draw.circle(self.screen, self.BLACK, (badge_x, badge_y), 12, 1)
                
                queue_text = self.font_small.render(str(len(driver.order_queue)), 
                                                    True, self.BLACK)
                queue_rect = queue_text.get_rect(center=(badge_x, badge_y))
                self.screen.blit(queue_text, queue_rect)
    
    def draw_orders(self):
        """Draw pending order destinations"""
        for order_id in self.env.pending_orders:
            order = self.env.orders[order_id]
            x, y = self.node_to_pos(order.destination)
            
            # Pulsing effect (optional, looks cool)
            pulse = abs((pygame.time.get_ticks() // 300) % 10 - 5)
            size = 15 + pulse
            
            # Draw X marker
            pygame.draw.line(self.screen, self.ORANGE, 
                           (x - size, y - size), (x + size, y + size), 4)
            pygame.draw.line(self.screen, self.ORANGE, 
                           (x - size, y + size), (x + size, y - size), 4)
    
    def draw_stats(self):
        """Draw statistics panel"""
        stats_y = self.height - 140
        
        # Background panel
        pygame.draw.rect(self.screen, self.LIGHT_GRAY, 
                        (10, stats_y, self.width - 20, 130))
        pygame.draw.rect(self.screen, self.DARK_GRAY, 
                        (10, stats_y, self.width - 20, 130), 2)
        
        # Title
        title = self.font_large.render('SIMULATION STATUS', True, self.BLACK)
        self.screen.blit(title, (20, stats_y + 10))
        
        # Stats
        stats = [
            f"Time: {self.env.current_time} minutes",
            f"Completed Orders: {len(self.env.completed_orders)}",
            f"Pending Orders: {len(self.env.pending_orders)}",
            f"Speed: {self.speed} steps/sec"
        ]
        
        y_offset = stats_y + 45
        for i, stat in enumerate(stats):
            x_pos = 20 if i < 2 else self.width // 2 + 20
            if i >= 2:
                y_offset_adjusted = stats_y + 45 + (i - 2) * 25
            else:
                y_offset_adjusted = y_offset + i * 25
            
            text = self.font_medium.render(stat, True, self.BLACK)
            self.screen.blit(text, (x_pos, y_offset_adjusted))
        
        # Controls hint
        if self.paused:
            pause_text = self.font_large.render('PAUSED', True, (200, 0, 0))
            pause_rect = pause_text.get_rect(center=(self.width // 2, stats_y + 100))
            self.screen.blit(pause_text, pause_rect)
        else:
            controls = self.font_small.render(
                'SPACE: Pause | +/-: Speed | ESC: Quit', 
                True, self.DARK_GRAY)
            controls_rect = controls.get_rect(center=(self.width // 2, stats_y + 105))
            self.screen.blit(controls, controls_rect)
        
        # Driver legend
        legend_x = 20
        legend_y = stats_y + 100
        for i in range(3):
            color = self.DRIVER_COLORS[i]
            pygame.draw.circle(self.screen, color, (legend_x + i * 80, legend_y), 10)
            text = self.font_small.render(f'Driver {i}', True, self.BLACK)
            self.screen.blit(text, (legend_x + i * 80 + 15, legend_y - 7))
    
    def run(self):
        """Main visualization loop"""
        self.env.reset()
        step = 0
        max_steps = 500
        
        print("Pygame window opened!")
        print("Controls:")
        print("  SPACE - Pause/Resume")
        print("  + - Speed up")
        print("  - - Slow down")
        print("  ESC - Quit")
        
        while self.running and step < max_steps:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                        print("PAUSED" if self.paused else "RESUMED")
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self.speed = min(20, self.speed + 1)
                        print(f"Speed: {self.speed} steps/sec")
                    elif event.key == pygame.K_MINUS:
                        self.speed = max(1, self.speed - 1)
                        print(f"Speed: {self.speed} steps/sec")
            
            # Simulation step (if not paused)
            if not self.paused:
                # Random order assignment (baseline - RL will replace this)
                if self.env.pending_orders:
                    order_id = self.env.pending_orders[0]
                    driver_id = random.randint(0, self.env.num_drivers - 1)
                    position = len(self.env.drivers[driver_id].order_queue)
                    self.env.assign_order(order_id, driver_id, position)
                
                state, reward, done = self.env.step()
                step += 1
                
                if done:
                    print(f"\nSimulation completed at step {step}")
                    self.running = False
            
            # Draw everything
            self.screen.fill(self.WHITE)
            self.draw_grid()
            self.draw_orders()
            self.draw_drivers()
            self.draw_stats()
            
            pygame.display.flip()
            self.clock.tick(self.speed)
        
        # Show final stats before closing
        print("\n" + "="*50)
        print("FINAL STATISTICS")
        print("="*50)
        completed = len(self.env.completed_orders)
        if completed > 0:
            late = sum(1 for oid in self.env.completed_orders 
                      if self.env.orders[oid].delivery_time > self.env.orders[oid].deadline)
            avg_time = sum(self.env.orders[oid].delivery_time - self.env.orders[oid].arrival_time 
                          for oid in self.env.completed_orders) / completed
            print(f"Completed orders: {completed}")
            print(f"Late orders: {late}")
            print(f"On-time rate: {100*(completed-late)/completed:.1f}%")
            print(f"Average delivery time: {avg_time:.1f} minutes")
        print("="*50)
        
        pygame.quit()

if __name__ == "__main__":
    print("Initializing Delivery Optimization Simulation...")
    
    env = DeliveryEnvironment(
        num_drivers=3,
        grid_size=5,
        traffic_pattern='rush_hour',
        order_arrival_rate=0.3
    )
    
    print(f"Environment created: {env.graph.num_nodes} nodes, {len(env.graph.edges)} edges")
    print("Starting visualization...\n")
    
    viz = DeliveryVisualizer(env)
    viz.run()