#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cupti_events.h>
#include <unistd.h>


int main(int argc, char *argv[])
{
  CUcontext context;
  CUdevice device;

  CUptiResult cuptiErr;
  CUpti_EventGroup eventGroup;
  CUpti_EventID eventId;

  const char *eventName;

  size_t bytesRead, valueSize;
  uint32_t numInstances = 0, j = 0;
  uint64_t *eventValues = NULL, eventVal = 0;
  uint32_t profile_all = 1;


  if (argc > 1)
    deviceNum = atoi(argv[1]);
  else
    deviceNum = 0;

  if (argc > 2) {
    eventName = argv[2];
  }
  else {
    eventName = "inst_executed";
  }

  
  cuDeviceGet(&device, deviceNum);

  cuCtxCreate(&context, 0, device);


  cuptiSetEventCollectionMode(context, CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS);


  cuptiEventGroupCreate(context, &eventGroup, 0);


  cuptiEventGetIdFromName(device, eventName, &eventId);
  

  cuptiEventGroupAddEvent(eventGroup, eventId);
  

  cuptiEventGroupSetAttribute(eventGroup, CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(profile_all), &profile_all);
  
  cuptiErr = cuptiEventGroupEnable(eventGroup);
 
  valueSize = sizeof(numInstances);
  cuptiErr = cuptiEventGroupGetAttribute(eventGroup, CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT, &valueSize, &numInstances);

  bytesRead = sizeof(uint64_t) * numInstances;
  eventValues = (uint64_t *) malloc(bytesRead);

  if (eventValues == NULL) {
      printf("%s:%d: Failed to allocate memory.\n", __FILE__, __LINE__);
      exit(-1);
  }

  int i = 0;

  do {
    cuptiErr = cuptiEventGroupReadEvent(eventGroup,
                                        CUPTI_EVENT_READ_FLAG_NONE,
                                        eventId, &bytesRead, eventValues);
  
    if (bytesRead != (sizeof(uint64_t) * numInstances)) {
      printf("Failed to read value for \"%s\"\n", eventName);
      exit(-1);
    }

    for (j = 0; j < numInstances; j++) {
      eventVal += eventValues[j];
    }
    printf("%s: %llu\n", eventName, (unsigned long long)eventVal);
    sleep(1);
    i+=1;
  } while (i<=30);

  cuptiErr = cuptiEventGroupDisable(eventGroup);

  cuptiErr = cuptiEventGroupDestroy(eventGroup);

  free(eventValues);
  return 0;
}
