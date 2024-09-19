import numpy as np
import matplotlib.pyplot as plt

# Define the x values
x = np.linspace(-10, 10, 400)

# Define the functions p(x), q(x), and g(x)
p_x = np.exp(-x**2)  # Gaussian for p(x)
q_x = np.exp(-(x - 2)**2) * 0.5  # Gaussian for q(x), shifted and scaled
g_x = np.sin(x) * np.exp(-0.1 * x**2)  # Damped sine wave for g(x)

# Create the plot
plt.figure(figsize=(10, 6))
# plt.plot(x, p_x, label='p(x)', color='black', linewidth=10)
# plt.plot(x, q_x, label='q(x)', color='red', linestyle='--', linewidth=10)
plt.plot(x, g_x, label='g(x)', color='black', linestyle='-', linewidth=10)

# Add vertical lines for S+ and S-
# plt.axvline(x=-2, color='gray', linestyle='--', label='S+')
# plt.axvline(x=2, color='gray', linestyle='--', label='S-')
plt.axhline(y=0, color='gray', linestyle='-')

# # Add annotations
# plt.text(-3, 0.1, 'Composition', fontsize=10, horizontalalignment='center')
# plt.text(0, -0.1, 'Augmenting', fontsize=10, horizontalalignment='center')
# plt.text(3, -0.1, 'Filtering', fontsize=10, horizontalalignment='center')

# Set labels and title
plt.title('Generated Curves')
plt.xlabel('x')
plt.ylabel('Density')

plt.ylim(-0.2, 1.1)
plt.xlim(-15, 10)

# Show the plot
plt.savefig("draw/curves_sir.pdf", bbox_inches='tight', pad_inches=0.1,format="pdf")