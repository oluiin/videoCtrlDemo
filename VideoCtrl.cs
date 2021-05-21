using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Video; // Video 사용하기 위해 선언
using UnityEngine.UI; // UI 사용하기 위해 선언
using System.Threading;
using System.Net;
using System.Net.Sockets;
using System.Text;

public class VideoCtrl : MonoBehaviour
{
    public Image image;
    public Sprite[] sprites;
    public VideoPlayer video;

    //통신을 위한
    Thread mThread;
    public string connectionIP = "127.0.0.1";
    public int connectionPort = 9090;
    IPAddress localAdd;
    TcpListener listener;
    TcpClient client;
    
    int recNum = 0;

    bool running;
    //통신 선언 끝

    [Header("숫자 변수")]
    public int numberValue;

    [Header("이미지가 서서히 사라지는 데에 걸리는 시간")]
    public float time;

    [Header("이미지가 사라지 전 띄워둘 시간")]
    public float delay;

    [Header("볼륨 조절 정도")]
    public float volumeValue;

    int currentNumber;
    bool isImageOn;

    void Update()
    {
        numberValue = recNum;
        // 입력한 숫자가 있고, 그 숫자가 이전과 달라졌다면
        if (numberValue != currentNumber)
        {
            // 기능 시작
            NumberCtrl();
        }

        // 지금 값을 이전 값으로 저장
        currentNumber = numberValue;

        // 이미지 한 번에 보이게
        if (isImageOn)
        {
            image.color = Color.white;
        }

        // 띄운 이미지 서서히 안 보이게
        else
        {
            image.color = Vector4.Lerp(image.color, new Vector4(1, 1, 1, 0), time * Time.deltaTime);
        }
    }

    private void Start()
    {
        ThreadStart ts = new ThreadStart(GetInfo);
        mThread = new Thread(ts);
        mThread.Start();
    }

    // 기능 시작
    void NumberCtrl()
    {
        switch (numberValue)
        {
            // 재생
            case 0:
                // 이미지 띄우기
                isImageOn = true;

                // 비디오 재생
                video.Play();

                // 이미지 지정
                image.sprite = sprites[0];

                // 1초 후 호출
                Invoke("ImageDelay", delay);
                break;
            // 일시정지
            case 1:
                // 이미지 띄우기
                isImageOn = true;

                // 비디오 일시정지
                video.Pause();

                // 이미지 지정
                image.sprite = sprites[1];

                // 1초 후 호출
                Invoke("ImageDelay", delay);
                break;
            // 볼륨 업
            case 2:
                // 이미지 띄우기
                isImageOn = true;

                // 볼륨 조금씩 키우기 (1보다 작을 때만)
                if (video.GetDirectAudioVolume(0) < 1)
                    video.SetDirectAudioVolume(0, video.GetDirectAudioVolume(0) + volumeValue);

                // 이미지 지정
                image.sprite = sprites[2];

                // 1초 후 호출
                Invoke("ImageDelay", delay);
                break;
            // 볼륨 다운
            case 3:
                // 이미지 띄우기
                isImageOn = true;

                // 볼륨 조금씩 줄이기 (0보다 클 때만)
                if (video.GetDirectAudioVolume(0) > 0)
                    video.SetDirectAudioVolume(0, video.GetDirectAudioVolume(0) - volumeValue);

                // 이미지 지정
                image.sprite = sprites[3];

                // 1초 후 호출
                Invoke("ImageDelay", delay);
                break;
            // 볼륨 max
            case 4:
                // 이미지 띄우기
                isImageOn = true;
                int volumeMax = 1;
                // 볼륨 조금씩 줄이기 (0보다 클 때만)
                if (video.GetDirectAudioVolume(0) < 1)
                    video.SetDirectAudioVolume(0, volumeMax);

                // 이미지 지정
                image.sprite = sprites[4];

                // 1초 후 호출
                Invoke("ImageDelay", delay);
                break;
            //볼륨 zero
            case 5:
                // 이미지 띄우기
                isImageOn = true;
                int volumeMin = 0;
                
                video.SetDirectAudioVolume(0, volumeMin);

                // 이미지 지정
                image.sprite = sprites[5];

                // 1초 후 호출
                Invoke("ImageDelay", delay);
                break;

        }
    }

    // 딜레이 후 이미지 서서히 안 보이게
    void ImageDelay()
    {
        isImageOn = false;
    }

    void GetInfo()
    {
        localAdd = IPAddress.Parse(connectionIP);
        listener = new TcpListener(IPAddress.Any, connectionPort);
        listener.Start();

        client = listener.AcceptTcpClient();

        running = true;
        while (running)
        {
            SendAndReceiveData();
        }
        listener.Stop();
    }

    void SendAndReceiveData()
    {
        NetworkStream nwStream = client.GetStream();
        byte[] buffer = new byte[client.ReceiveBufferSize];

        //---receiving Data from the Host----
        int bytesRead = nwStream.Read(buffer, 0, client.ReceiveBufferSize); //Getting data in Bytes from Python
        string dataReceived = Encoding.UTF8.GetString(buffer, 0, bytesRead); //Converting byte data to string

        if (dataReceived != null)
        {
            //---Using received data---
            recNum = StringToInt(dataReceived); //<-- assigning recNum value from Python
            print("received pos data, and moved the Cube!");

            //---Sending Data to Host----
            byte[] myWriteBuffer = Encoding.ASCII.GetBytes("Hey I got your message Python! Do You see this massage?"); //Converting string to byte data
            nwStream.Write(myWriteBuffer, 0, myWriteBuffer.Length); //Sending the data in Bytes to Python
        }
    }

    public static int StringToInt(string str)
    {
        // Remove the parentheses

        int result = int.Parse(str);

        return result;
    }
}

 