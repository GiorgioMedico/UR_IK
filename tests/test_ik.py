# import numpy as np
# import time
# from ik import forward_kinematics, inverse_kinematics, is_singularity


# def test_ik_with_random_angles(num_tests=100, angle_range=(-np.pi, np.pi), epsilon=1e-4, dh_params=None):
#     """
#     Test inverse kinematics algorithm with random joint angles.

#     Parameters:
#     num_tests: Number of random tests to perform
#     angle_range: Range for random joint angles (min, max)
#     epsilon: Threshold for considering errors negligible
#     dh_params: Dictionary with DH parameters (if None, default params are used)

#     Returns:
#     Dictionary containing test statistics
#     """
#     # Use default DH parameters if none provided
#     if dh_params is None:
#         dh_params = {
#             'a2': -0.425,  # Link length (m)
#             'a3': -0.3922,  # Link length (m)
#             'd1': 0.1625,   # Link offset (m)
#             'd4': 0.1333,   # Link offset (m)
#             'd5': 0.0997,   # Link offset (m)
#             'd6': 0.0996    # Link offset (m)
#         }

#     # Statistics counters
#     successful_tests = 0
#     failed_tests = 0

#     # Error tracking
#     position_errors = []
#     orientation_errors = []
#     angle_errors = []
#     solutions_count = []
#     execution_times = []

#     print(f"Running {num_tests} random IK tests...")

#     for test_idx in range(num_tests):
#         try:
#             # 1. Generate random joint angles within specified range
#             random_angles = np.random.uniform(angle_range[0], angle_range[1], 6)

#             # 2. Compute forward kinematics
#             T06, _ = forward_kinematics(random_angles, dh_params)

#             # 3. Record start time for performance measurement
#             start_time = time.time()

#             # 4. Compute inverse kinematics
#             ik_solutions = inverse_kinematics(T06, dh_params)

#             # 5. Record execution time
#             execution_time = time.time() - start_time
#             execution_times.append(execution_time)

#             # 6. Track number of solutions found
#             num_solutions = len(ik_solutions)
#             solutions_count.append(num_solutions)

#             if num_solutions == 0:
#                 print(f"Test {test_idx+1}: No solutions found")
#                 failed_tests += 1
#                 continue

#             # 7. Validate all solutions
#             valid_solutions = 0
#             best_solution = None
#             min_angle_error = float('inf')

#             for solution in ik_solutions:
#                 # Verify solution with forward kinematics
#                 T06_verify, _ = forward_kinematics(solution, dh_params)

#                 # Compute errors
#                 pos_error = np.linalg.norm(T06[0:3, 3] - T06_verify[0:3, 3])
#                 ori_error = np.linalg.norm(T06[0:3, 0:3] - T06_verify[0:3, 0:3], 'fro')

#                 # Check if solution is valid
#                 if pos_error < epsilon and ori_error < epsilon:
#                     valid_solutions += 1

#                     # Calculate joint angle error
#                     angle_error = np.linalg.norm(random_angles - solution)

#                     if angle_error < min_angle_error:
#                         min_angle_error = angle_error
#                         best_solution = solution

#             # If we found valid solutions
#             if valid_solutions > 0:
#                 # Record errors for the best solution
#                 T06_best, _ = forward_kinematics(best_solution, dh_params)
#                 pos_error = np.linalg.norm(T06[0:3, 3] - T06_best[0:3, 3])
#                 ori_error = np.linalg.norm(T06[0:3, 0:3] - T06_best[0:3, 0:3], 'fro')

#                 position_errors.append(pos_error)
#                 orientation_errors.append(ori_error)
#                 angle_errors.append(min_angle_error)

#                 # Update success counter if all solutions are valid
#                 if valid_solutions == num_solutions:
#                     successful_tests += 1
#                 else:
#                     print(f"Test {test_idx+1}: {valid_solutions}/{num_solutions} valid solutions")
#                     failed_tests += 1
#             else:
#                 print(f"Test {test_idx+1}: No valid solutions among {num_solutions} candidates")
#                 failed_tests += 1

#             # 8. Print progress every 10 tests
#             if (test_idx + 1) % 10 == 0:
#                 print(f"Completed {test_idx + 1} tests. Success rate: {successful_tests/(test_idx+1)*100:.2f}%")

#         except Exception as e:
#             print(f"Test {test_idx+1} failed with error: {str(e)}")
#             failed_tests += 1

#     # Calculate test statistics
#     success_rate = successful_tests / num_tests * 100
#     avg_pos_error = np.mean(position_errors) if position_errors else float('nan')
#     avg_ori_error = np.mean(orientation_errors) if orientation_errors else float('nan')
#     avg_angle_error = np.mean(angle_errors) if angle_errors else float('nan')
#     avg_solutions = np.mean(solutions_count) if solutions_count else float('nan')
#     avg_exec_time = np.mean(execution_times) if execution_times else float('nan')

#     # Create result dictionary
#     results = {
#         'success_rate': success_rate,
#         'successful_tests': successful_tests,
#         'failed_tests': failed_tests,
#         'avg_position_error': avg_pos_error,
#         'avg_orientation_error': avg_ori_error,
#         'avg_angle_error': avg_angle_error,
#         'avg_solutions_count': avg_solutions,
#         'max_position_error': max(position_errors) if position_errors else float('nan'),
#         'max_orientation_error': max(orientation_errors) if orientation_errors else float('nan'),
#         'avg_execution_time': avg_exec_time,
#         'max_execution_time': max(execution_times) if execution_times else float('nan')
#     }

#     return results


# def test_singularity_cases(epsilon=1e-4, verbose=True):
#     """
#     Test inverse kinematics with configurations near singularities.
#     Tests all three types of singularities:
#     1. Wrist singularity (q5 ≈ 0)
#     2. Elbow singularity (q3 ≈ 0)
#     3. Shoulder singularity

#     Parameters:
#     epsilon: Threshold for considering errors negligible
#     verbose: Whether to print detailed output

#     Returns:
#     Dictionary containing test results
#     """
#     # Define DH parameters
#     dh_params = {
#         'a2': -0.5,  # Link length (m)
#         'a3': -0.4,  # Link length (m)
#         'd1': 0.1,   # Link offset (m)
#         'd4': 0.1,   # Link offset (m)
#         'd5': 0.1,   # Link offset (m)
#         'd6': 0.08   # Link offset (m)
#     }

#     # Test cases with different distances from singularity
#     singularity_distances = [1e-1, 1e-2, 1e-3, 1e-4]
#     test_results = {
#         'wrist': {dist: {} for dist in singularity_distances},
#         'elbow': {dist: {} for dist in singularity_distances},
#         'shoulder': {dist: {} for dist in singularity_distances}
#     }

#     # Test 1: Wrist Singularity (q5 ≈ 0)
#     if verbose:
#         print("\n=== Testing Wrist Singularity ===")

#     for dist in singularity_distances:
#         if verbose:
#             print(f"\nWrist singularity distance: {dist}")

#         # Define joint angles near wrist singularity
#         joint_angles = [0, np.pi/2, np.pi/2, 0, dist, 0]

#         # Compute forward kinematics
#         T06, _ = forward_kinematics(joint_angles, dh_params)

#         # Check if it's a singularity
#         is_sing = is_singularity(joint_angles[1], joint_angles[2], joint_angles[3], joint_angles[4], dh_params)

#         if verbose:
#             print(f"Is singularity: {is_sing}")

#         # Compute inverse kinematics
#         start_time = time.time()
#         solutions = inverse_kinematics(T06, dh_params, verbose=verbose)
#         execution_time = time.time() - start_time

#         # Record results
#         test_results['wrist'][dist] = {
#             'is_singularity': is_sing,
#             'num_solutions': len(solutions),
#             'execution_time': execution_time
#         }

#     # Test 2: Elbow Singularity (q3 ≈ 0)
#     if verbose:
#         print("\n=== Testing Elbow Singularity ===")

#     for dist in singularity_distances:
#         if verbose:
#             print(f"\nElbow singularity distance: {dist}")

#         # Define joint angles near elbow singularity
#         joint_angles = [0, np.pi/2, dist, 0, np.pi/2, 0]

#         # Compute forward kinematics
#         T06, _ = forward_kinematics(joint_angles, dh_params)

#         # Check if it's a singularity
#         is_sing = is_singularity(joint_angles[1], joint_angles[2], joint_angles[3], joint_angles[4], dh_params)

#         if verbose:
#             print(f"Is singularity: {is_sing}")

#         # Compute inverse kinematics
#         start_time = time.time()
#         solutions = inverse_kinematics(T06, dh_params, verbose=verbose)
#         execution_time = time.time() - start_time

#         # Record results
#         test_results['elbow'][dist] = {
#             'is_singularity': is_sing,
#             'num_solutions': len(solutions),
#             'execution_time': execution_time
#         }

#     # Test 3: Shoulder Singularity
#     if verbose:
#         print("\n=== Testing Shoulder Singularity ===")

#     # We need to construct a configuration that's close to a shoulder singularity
#     # For a 6-DOF robot, the shoulder singularity occurs when:
#     # c2*a2 + c23*a3 + s234*d5 ≈ 0

#     # Start with a configuration
#     base_angles = [0, 0, 0, 0, np.pi/2, 0]

#     # Adjust q2, q3, q4 to get close to shoulder singularity
#     # This requires some trial and error or mathematical derivation
#     # For simplicity, we'll adjust q2 and check when we get close to singularity

#     for dist in singularity_distances:
#         if verbose:
#             print(f"\nShoulder singularity approximate distance: {dist}")

#         # Use numerical approach to find a configuration close to shoulder singularity
#         found_config = False
#         for q2_test in np.linspace(-np.pi, np.pi, 100):
#             for q3_test in np.linspace(-np.pi, np.pi, 10):
#                 for q4_test in np.linspace(-np.pi, np.pi, 10):
#                     test_config = [0, q2_test, q3_test, q4_test, np.pi/2, 0]

#                     # Calculate shoulder singularity measure
#                     c2 = np.cos(q2_test)
#                     c23 = np.cos(q2_test + q3_test)
#                     s234 = np.sin(q2_test + q3_test + q4_test)

#                     measure = c2 * dh_params['a2'] + c23 * dh_params['a3'] + s234 * dh_params['d5']

#                     if abs(measure) < dist * 10:  # We multiply by 10 to relax the constraint a bit
#                         joint_angles = test_config
#                         found_config = True
#                         break
#                 if found_config:
#                     break
#             if found_config:
#                 break

#         if not found_config:
#             if verbose:
#                 print(f"Could not find configuration within {dist} of shoulder singularity")
#             continue

#         # Compute forward kinematics
#         T06, _ = forward_kinematics(joint_angles, dh_params)

#         # Check if it's a singularity
#         is_sing = is_singularity(joint_angles[1], joint_angles[2], joint_angles[3], joint_angles[4], dh_params)

#         if verbose:
#             print(f"Joint angles: {[round(a, 4) for a in joint_angles]}")
#             print(f"Is singularity: {is_sing}")

#             # Calculate and print the exact measure to verify
#             c2 = np.cos(joint_angles[1])
#             c23 = np.cos(joint_angles[1] + joint_angles[2])
#             s234 = np.sin(joint_angles[1] + joint_angles[2] + joint_angles[3])
#             measure = c2 * dh_params['a2'] + c23 * dh_params['a3'] + s234 * dh_params['d5']
#             print(f"Shoulder singularity measure: {measure:.6f}")

#         # Compute inverse kinematics
#         start_time = time.time()
#         solutions = inverse_kinematics(T06, dh_params, verbose=verbose)
#         execution_time = time.time() - start_time

#         # Record results
#         test_results['shoulder'][dist] = {
#             'is_singularity': is_sing,
#             'num_solutions': len(solutions),
#             'execution_time': execution_time,
#             'joint_angles': joint_angles,
#             'singularity_measure': measure
#         }

#     return test_results


# def test_unreachable_positions(verbose=True):
#     """
#     Test inverse kinematics with unreachable end-effector positions.

#     Parameters:
#     verbose: Whether to print detailed output

#     Returns:
#     Dictionary containing test results
#     """
#     # Define DH parameters
#     dh_params = {
#         'a2': -0.5,  # Link length (m)
#         'a3': -0.4,  # Link length (m)
#         'd1': 0.1,   # Link offset (m)
#         'd4': 0.1,   # Link offset (m)
#         'd5': 0.1,   # Link offset (m)
#         'd6': 0.08   # Link offset (m)
#     }

#     # Calculate maximum reach
#     max_reach = abs(dh_params['a2']) + abs(dh_params['a3']) + abs(dh_params['d5']) + abs(dh_params['d6'])

#     if verbose:
#         print(f"\n=== Testing Unreachable Positions ===")
#         print(f"Maximum theoretical reach: {max_reach:.4f} m")

#     # Test cases with positions just outside reachable workspace
#     test_results = []

#     # Test different distances beyond maximum reach
#     for distance_factor in [1.01, 1.1, 1.2, 1.5, 2.0]:
#         test_distance = max_reach * distance_factor

#         if verbose:
#             print(f"\nTesting position at {distance_factor:.2f} times max reach ({test_distance:.4f} m)")

#         # Create transformation matrix with position outside reachable workspace
#         # Start with identity matrix
#         T = np.eye(4)

#         # Set position in x direction
#         T[0, 3] = test_distance

#         # Try to compute inverse kinematics
#         start_time = time.time()
#         solutions = inverse_kinematics(T, dh_params, verbose=verbose)
#         execution_time = time.time() - start_time

#         result = {
#             'distance_factor': distance_factor,
#             'position': T[0:3, 3],
#             'num_solutions': len(solutions),
#             'execution_time': execution_time
#         }

#         test_results.append(result)

#         if verbose:
#             print(f"Number of solutions: {len(solutions)}")
#             print(f"Execution time: {execution_time:.4f} s")

#     return test_results


# def test_multiple_dh_params(num_tests=50, verbose=True):
#     """
#     Test inverse kinematics with different robot geometries.

#     Parameters:
#     num_tests: Number of tests per robot geometry
#     verbose: Whether to print detailed output

#     Returns:
#     Dictionary containing test results for each robot geometry
#     """
#     # Define different robot geometries (DH parameters)
#     robot_geometries = {
#         'UR5': {
#             'a2': -0.425,     # Link length (m)
#             'a3': -0.3922,    # Link length (m)
#             'd1': 0.1625,     # Link offset (m)
#             'd4': 0.1333,     # Link offset (m)
#             'd5': 0.0997,     # Link offset (m)
#             'd6': 0.0996      # Link offset (m)
#         },
#         'KUKA_KR6': {
#             'a2': -0.455,     # Link length (m)
#             'a3': -0.42,      # Link length (m)
#             'd1': 0.4,        # Link offset (m)
#             'd4': 0.42,       # Link offset (m)
#             'd5': 0.08,       # Link offset (m)
#             'd6': 0.08        # Link offset (m)
#         },
#         'SMALL_ROBOT': {
#             'a2': -0.25,      # Link length (m)
#             'a3': -0.2,       # Link length (m)
#             'd1': 0.05,       # Link offset (m)
#             'd4': 0.05,       # Link offset (m)
#             'd5': 0.05,       # Link offset (m)
#             'd6': 0.04        # Link offset (m)
#         },
#         'LARGE_ROBOT': {
#             'a2': -1.0,       # Link length (m)
#             'a3': -0.8,       # Link length (m)
#             'd1': 0.2,        # Link offset (m)
#             'd4': 0.2,        # Link offset (m)
#             'd5': 0.2,        # Link offset (m)
#             'd6': 0.16        # Link offset (m)
#         }
#     }

#     # Results dictionary
#     results = {}

#     for robot_name, dh_params in robot_geometries.items():
#         if verbose:
#             print(f"\n=== Testing Robot Geometry: {robot_name} ===")

#         # Run tests with this robot geometry
#         test_results = test_ik_with_random_angles(num_tests=num_tests, dh_params=dh_params)

#         # Store results
#         results[robot_name] = test_results

#         # Print summary
#         if verbose:
#             print(f"Success Rate: {test_results['success_rate']:.2f}%")
#             print(f"Average Position Error: {test_results['avg_position_error']:.6f} m")
#             print(f"Average Orientation Error: {test_results['avg_orientation_error']:.6f}")
#             print(f"Average Number of Solutions: {test_results['avg_solutions_count']:.2f}")
#             print(f"Average Execution Time: {test_results['avg_execution_time']:.4f} s")

#     return results


# def run_full_test_suite():
#     """
#     Run a comprehensive test suite for the inverse kinematics implementation.
#     """
#     print("===== INVERSE KINEMATICS COMPREHENSIVE TEST SUITE =====\n")

#     # 1. Random angles test
#     print("\n1. RANDOM CONFIGURATIONS TEST")
#     print("-----------------------------")
#     results = test_ik_with_random_angles(num_tests=10000)

#     print("\nRandom Test Results:")
#     print(f"Success Rate: {results['success_rate']:.2f}%")
#     print(f"Successful Tests: {results['successful_tests']}")
#     print(f"Failed Tests: {results['failed_tests']}")
#     print(f"Average Position Error: {results['avg_position_error']:.6f} m")
#     print(f"Average Orientation Error: {results['avg_orientation_error']:.6f}")
#     print(f"Average Joint Angle Error: {results['avg_angle_error']:.6f} rad")
#     print(f"Average Number of Solutions: {results['avg_solutions_count']:.2f}")
#     print(f"Average Execution Time: {results['avg_execution_time']:.4f} s")

#     # 2. Singularity test
#     print("\n2. SINGULARITY TEST")
#     print("------------------")
#     singularity_results = test_singularity_cases(verbose=False)

#     print("\nSingularity Test Results:")
#     for sing_type in singularity_results:
#         print(f"\n{sing_type.capitalize()} Singularity:")
#         for dist, result in singularity_results[sing_type].items():
#             if 'num_solutions' in result:
#                 print(f"  Distance {dist}: {result['num_solutions']} solutions, " +
#                       f"is_singularity={result['is_singularity']}, " +
#                       f"time={result['execution_time']:.4f}s")

#     # 3. Unreachable positions test
#     print("\n3. UNREACHABLE POSITIONS TEST")
#     print("----------------------------")
#     unreachable_results = test_unreachable_positions(verbose=False)

#     print("\nUnreachable Positions Test Results:")
#     for result in unreachable_results:
#         print(f"  Position at {result['distance_factor']:.2f}x max reach: " +
#               f"{result['num_solutions']} solutions, " +
#               f"time={result['execution_time']:.4f}s")

#     # 4. Multiple robot geometries test
#     print("\n4. MULTIPLE ROBOT GEOMETRIES TEST")
#     print("--------------------------------")
#     geometry_results = test_multiple_dh_params(num_tests=50, verbose=False)

#     print("\nMultiple Robot Geometries Test Results:")
#     for robot, result in geometry_results.items():
#         print(f"\n{robot}:")
#         print(f"  Success Rate: {result['success_rate']:.2f}%")
#         print(f"  Avg. Position Error: {result['avg_position_error']:.6f} m")
#         print(f"  Avg. Number of Solutions: {result['avg_solutions_count']:.2f}")
#         print(f"  Avg. Execution Time: {result['avg_execution_time']:.4f} s")

#     print("\n===== TEST SUITE COMPLETED =====")


# if __name__ == "__main__":
#     # Run the full test suite
#     run_full_test_suite()
