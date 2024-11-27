import numpy as np 
import cv2
import OptFlow as OF

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 500

feature_params = dict(maxCorners=500,
                      qualityLevel=0.3,
                      minDistance=10,
                      blockSize=7)

lk_params = dict(winSize  = (21, 21), 
				maxLevel = 3,
             	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def featureTracking(image_ref, image_cur, px_ref):
	kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]

	st = st.reshape(st.shape[0])
	kp1 = px_ref[st == 1]
	kp2 = kp2[st == 1]

	return kp1, kp2


class PinholeCamera:
	def __init__(self, width, height, fx, fy, cx, cy, 
				k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
		self.width = width
		self.height = height
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		self.distortion = (abs(k1) > 0.0000001)
		self.d = [k1, k2, p1, p2, k3]


class VisualOdometry:
	def __init__(self, cam):
		self.frame_stage = 0
		self.cam = cam
		self.new_frame = None
		self.last_frame = None
		self.cur_R = None
		self.cur_t = None
		self.px_ref = None
		self.px_cur = None
		self.Scale = None
		self.skip_frame = False
		self.focal = cam.fx
		self.T_vectors = []
		self.R_matrices = []
		self.new_cloud = None
		self.last_cloud = None
		self.K = np.array([[cam.fx,0,cam.cx],[0,cam.fy,cam.cy],[0,0,1]])
		self.pp = (cam.cx, cam.cy)
		self.trueX, self.trueY, self.trueZ = 0, 0, 0
		self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
  
		self.acc = np.zeros((3,1))
		self.Rot = np.zeros((3,3))
		#with open(annotations) as f:
		#	self.annotations = f.readlines()
	
	def detectNewFeatures(self, cur_img):
		"""Detects new features in the current frame.
        Uses the Feature Detector selected."""
		if self.detector == 'SHI-TOMASI':
			feature_pts = cv2.goodFeaturesToTrack(cur_img, **feature_params)
			feature_pts = np.array([x for x in feature_pts], dtype=np.float32).reshape((-1, 2))
		else:
			feature_pts = self.detector.detect(cur_img, None)
			feature_pts = np.array([x.pt for x in feature_pts], dtype=np.float32)

		return feature_pts
	
	def triangulatePoints(self, R, t):
		"""Triangulates the feature correspondence points with
        the camera intrinsic matrix, rotation matrix, and translation vector.
        It creates projection matrices for the triangulation process."""

        # The canonical matrix (set as the origin)
		P0 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]])
		P0 = self.K.dot(P0)
        # Rotated and translated using P0 as the reference point
		P1 = np.hstack((R, t))
		P1 = self.K.dot(P1)
        # Reshaped the point correspondence arrays to cv2.triangulatePoints's format
		point1 = self.px_ref.reshape(2, -1)
		point2 = self.px_cur.reshape(2, -1)

		return cv2.triangulatePoints(P0, P1, point1, point2).reshape(-1, 4)[:, :3]

	def getAbsoluteScale(self, frame_id):  #specialized for KITTI odometry dataset
		ss = self.annotations[frame_id-1].strip().split()
		x_prev = float(ss[3])
		y_prev = float(ss[7])
		z_prev = float(ss[11])
		ss = self.annotations[frame_id].strip().split()
		x = float(ss[3])
		y = float(ss[7])
		z = float(ss[11])
		self.trueX, self.trueY, self.trueZ = x, y, z
		return np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))
	
	def getRelativeScale(self):
		""" Returns the relative scale based on the 3-D point clouds
		produced by the triangulation_3D function. Using a pair of 3-D corresponding points
		the distance between them is calculated. This distance is then divided by the
		corresponding points' distance in another point cloud.
        """

		min_idx = min([self.new_cloud.shape[0], self.last_cloud.shape[0]])
		ratios = []  # List to obtain all the ratios of the distances
		for i in range(min_idx):
			if i > 0:
				Xk = self.new_cloud[i]
				p_Xk = self.new_cloud[i - 1]
				Xk_1 = self.last_cloud[i]
				p_Xk_1 = self.last_cloud[i - 1]

				if np.linalg.norm(p_Xk - Xk) != 0:
					ratios.append(np.linalg.norm(p_Xk_1 - Xk_1) / np.linalg.norm(p_Xk - Xk))

		d_ratio = np.median(ratios) # Take the median of ratios list as the final ratio
		return d_ratio

	def processFirstFrame(self):
		self.px_ref = self.detector.detect(self.new_frame)
		self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
		self.frame_stage = STAGE_SECOND_FRAME

	def processSecondFrame(self):
		self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
		E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)

		#if not E.isContinuous():
		if E is not None:
			E = E.copy()
			self.px_ref = self.px_ref.copy()
			self.px_cur = self.px_cur.copy()
		self.px_cur = np.ascontiguousarray(self.px_cur)
		self.px_ref = np.ascontiguousarray(self.px_ref)
		E = np.ascontiguousarray(E)
		_, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
		#_, R, t, mask = cv2.recoverPose(E, self.px_cur.get(), self.px_ref.get(), focal=self.focal, pp=self.pp)
		self.T_vectors.append(tuple(self.cur_R.dot(self.cur_t)))
		self.R_matrices.append(tuple(self.cur_R))
		self.new_cloud = self.triangulatePoints(self.cur_R, self.cur_t)
		self.OFF_prev, self.OFF_cur = self.px_ref, self.px_cur
		self.frame_stage = STAGE_DEFAULT_FRAME 
		self.px_ref = self.px_cur
		self.last_cloud = self.new_cloud

	def frame_Skip(self, pixel_diff):
		"""Determines if the current frame needs to be skipped.
         A frame is skipped on the basis that the current feature points
         are almost identical to the previous feature points, meaning the image
         was probably taken from the same place and the translation should be zero."""

        # We tried this parameter with 20, 15, 10, 5, 3, 2, 1 and 0
        # for one dataset and found that 3 produces the best results.
		return pixel_diff < 3
	
	def processFrame(self, frame_id):
		prev_img, cur_img = self.last_frame, self.new_frame

		#prev_img = cv2.undistort(prev_img, self.K)
		#cur_img = cv2.undistort()
		self.px_ref, self.px_cur, px_diff = OF.KLT_featureTracking(prev_img, cur_img, self.px_ref)
		#self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
		
		self.skip_frame = self.frame_Skip(px_diff)

		if self.skip_frame:
			if self.px_ref.shape[0] < kMinNumFeature:  # Verify if features on last_frame are sparse
				self.px_cur = self.detectNewFeatures(prev_img)
				self.px_ref = self.px_cur
				self.last_cloud = self.new_cloud
			return

		E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		_, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
		self.new_cloud = self.triangulatePoints(R, t)
		theta = np.deg2rad(-20)
		R_x = np.array([
		[1, 0, 0],
		[0, np.cos(theta), -np.sin(theta)],
		[0, np.sin(theta), np.cos(theta)]
		])
		t = R_x@t
		absolute_scale = self.getAbsoluteScale(frame_id)
		self.Scale = self.getRelativeScale()

		if (t[2] > t[0] and t[2] > t[1]):  # Accepts only dominant forward motion
			#self.cur_t = self.cur_t + self.Scale * self.cur_R.dot(t)  # Concatenate the translation vectors
			self.cur_t = self.cur_t + self.Scale * self.Rot.dot(t)
			self.cur_R = R.dot(self.cur_R)  # Concatenate the rotation matrix
			self.T_vectors.append(tuple(self.cur_t))
			self.R_matrices.append(tuple(self.cur_R))
		
		if self.px_ref.shape[0] < kMinNumFeature:                     # Verify if the amount of feature points
			self.px_cur = self.detectNewFeatures(cur_img)  # is above the kMinNumFeature threshold

		self.OFF_prev, self.OFF_cur = self.px_ref, self.px_cur
		self.px_ref = self.px_cur
		self.last_cloud = self.new_cloud

		if(absolute_scale > 0.1):
			self.cur_t = self.cur_t + absolute_scale*self.cur_R.dot(t) 
			self.cur_R = R.dot(self.cur_R)
		if(self.px_ref.shape[0] < kMinNumFeature):
			self.px_cur = self.detector.detect(self.new_frame)
			self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
		self.px_ref = self.px_cur


	def update(self, img, frame_id, data):
		assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
		flag = 0
		if data is not None:
			a = data.split()
			for i in range(len(a)):
				if a[i] is None:
					flag = 1
					break
			if flag == 0:
				
				print("mat", a)
				self.acc[0] = float(a[0].strip().strip(',[]array()'))
				self.acc[1] = float(a[1].strip().strip(',[]array()'))
				self.acc[2] = float(a[2].strip().strip(',[]array()'))
				print(self.acc)
			print("------------")
			for i in range(len(a)):
				print(a[i])
			print(a[5])
			print(a[6])
			self.Rot = np.zeros((3,3))
			self.Rot[0][0] = float(a[3].strip().strip(',[]array()'))
			self.Rot[0][1] = float(a[4].strip().strip(',[]array()'))
			self.Rot[0][2] = float(a[5].strip().strip(',[]array()'))
			idx = a[11].find(']')
			self.Rot[1][0] = float(a[6].strip().strip(',[]array()'))
			self.Rot[1][1] = float(a[7].strip().strip(',[]array()'))
			self.Rot[1][2] = float(a[8].strip().strip(',[]array()'))
			self.Rot[2][0] = float(a[9].strip().strip(',[]array()'))
			self.Rot[2][1] = float(a[10].strip().strip(',[]array()'))
			idx = a[11].find(']')
			self.Rot[2][2] = float(a[11][0:idx].strip().strip(',[]array()'))

		else:
			self.Abs = False
			self.Rot = self.cur_R

		self.new_frame = img
		self.new_roi = img[int(img.shape[0] * 0.40):img.shape[0], 0:img.shape[1]]
		if(self.frame_stage == STAGE_DEFAULT_FRAME):
			self.processFrame(frame_id)
		elif(self.frame_stage == STAGE_SECOND_FRAME):
			self.processSecondFrame()
		elif(self.frame_stage == STAGE_FIRST_FRAME):
			self.processFirstFrame()
		self.last_frame = self.new_frame

		if self.skip_frame: 
			return False
		
		self.last_id = frame_id
		self.last_frame = self.new_frame
		self.last_roi = self.new_roi

		return True