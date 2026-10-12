# แนวทางเชื่อม Hikvision IP Camera กับ Desktop

เอกสารนี้เป็นแผนสำหรับนักเรียนสาย network เท่านั้น คืนนี้ยังไม่ได้แก้ Windows Firewall, route, IP forwarding, NAT, VPN หรือ port ใด ๆ

## เป้าหมาย

```text
Hikvision Camera → PoE Switch → Camera Network → Desktop AI PC
                                           └→ หรือ Acer Gateway → private link → Desktop
Desktop: RTSP → OpenCV → MediaPipe → V4/Geometry → Dashboard
```

RTSP เป็นการรับ stream จากกล้อง ไม่ใช่การเปิด web server ของ project

## Option 1: route กล้องตรงไปยัง AI PC

```text
Camera → PoE switch → route/VLAN ที่อนุญาต → Desktop
```

ขั้นตอนในอนาคตคือกำหนด IP/subnet ของกล้องและ Desktop ตรวจ reachability จาก Desktop แล้วทดสอบ RTSP main/sub stream ด้วย diagnostic แบบจำกัดเวลา

ข้อดี: latency และจุดเสียหายน้อยกว่า, ไม่ต้องให้ Acer relay ตลอดเวลา

ข้อจำกัด: ต้องจัด routing/VLAN/สิทธิ์ใน network จริง และ Desktop ต้องเข้าถึง camera subnet ได้

## Option 2: Acer เป็น gateway หรือ relay

```text
Camera → PoE/Ethernet → Acer
                         ↓ private routed/VPN path หรือ local relay
                      Desktop AI PC
```

Acer ควรทำหน้าที่รับ/ส่ง network หรือ relay เท่านั้น ไม่ย้าย heavy AI ไปที่ Acer

ข้อดี: ซ่อน camera network จาก Desktop โดยตรงได้ และวาง Desktop อยู่อีก network ได้

ข้อจำกัด: เพิ่ม latency, bandwidth path, จุดเสียหาย และการดูแล process/secure link อีกชุดหนึ่ง

## เปรียบเทียบ

| แบบ | Latency | ความซับซ้อน | ความเสี่ยงจุดเสียหาย | เหมาะกับ |
|---|---|---|---|---|
| Direct routed | ต่ำกว่าโดยทั่วไป | กลาง | ต่ำกว่า | เมื่อ network อนุญาตให้ Desktop เข้าถึงกล้อง |
| Acer gateway/relay | เพิ่มตาม link/relay | สูงกว่า | Acer และ link เป็น dependency | เมื่อแยก camera network และ AI PC ไม่ได้ |
| VPN/private link | ขึ้นกับระบบที่เลือก | สูง | ขึ้นกับ tunnel/credentials | เมื่อสอง network อยู่คนละสถานที่หรือ policy แยกกัน |

ยังไม่เลือก VPN vendor และยังไม่ติดตั้ง relay server

## RTSP profile ที่เตรียมไว้

- `HIKVISION_MAIN` ใช้ path template `/Streaming/Channels/101`
- `HIKVISION_SUB` ใช้ path template `/Streaming/Channels/102`
- ไม่ hardcode camera IP, username หรือ password
- ค่าจริงอ่านจาก `.env` ที่ถูก ignore หรือ process environment

## ข้อห้ามด้านความปลอดภัย

- ห้าม port-forward RTSP ออก public Internet
- ห้ามใส่ password ใน source, batch file, report หรือ telemetry
- ห้ามเปิด port/listener โดยไม่ได้รับอนุญาต
- ห้ามแก้ firewall/route/IP forwarding/NAT ในงานเตรียมนี้
- ให้ใช้ private network และสิทธิ์เท่าที่จำเป็น
