package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import com.itextpdf.text.BaseColor;
import com.itextpdf.text.Chunk;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.pdf.PdfWriter;


public class SetChunkBackground {
	
	
	public static void main(String[] args){
		Document document = new Document();// 创建文档对象
		try {
			PdfWriter.getInstance(document, new FileOutputStream("c:\\Java编程全能词典.pdf"));// 关联文档对象与输出流
			document.open();// 打开文档
			Chunk chunk = new Chunk("Compile once, Run all!");// 定义块并添加内容
			chunk.setBackground(BaseColor.LIGHT_GRAY); // 设置背景颜色
			document.add(chunk);// 添加背景颜色
			document.close();// 关闭文档
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (DocumentException e) {
			e.printStackTrace();
		}
	}
}
