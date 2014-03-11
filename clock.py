

class Clock:
	def __init__(self,wakeTime,allowedError=30):
		self.wakeTime = wakeTime
		self.allowedError = allowedError
		#stage nos:
		#-1 - uninitialized
		#0 - Waking
		#1 - 1
		#2 - 2
		#3 - deep sleep
		#4 - REM
		self.priorStage = -1
		self.currStage = -1;
		
	def alarmSignal(self):
		currTime = getSystemTime;
		if(currTime - self.wakeTime<=self.allowedError):
			if((self.currStage == 1 || self.currStage == 2) and self.priorStage == 4):
				print('IDEAL WAKING')
				return True
			elif(self.currStage == 2 and self.priorStage == 1):
				print('OK WAKING')
				return True				
		elif(currTime > self.wakeTime):
			print('FAILURE TO WAKE')
			return True
		else:
			return False
	
	def update(self,signal):
		self.priorStage = self.currStage
		self.currStage = signal
		
		

#while(clock.alarmSignal):
	#PUT ANALYSIS LOOP HERE
	#clock.update(signal)