package com.zzk;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import com.itextpdf.text.Chunk;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Font;
import com.itextpdf.text.FontFactory;
import com.itextpdf.text.pdf.PdfWriter;

public class CreateAddChunk {
	
	
	public static void main(String args[]){
		Document document = new Document();	//创建文档对象
		try {
			PdfWriter.getInstance(document, new FileOutputStream("C:\\Java.pdf"));// 关联文档对象与输出流
			document.open();				//打开文档
			Chunk chunk1 = new Chunk("Text chunk1",FontFactory.getFont(FontFactory.COURIER_BOLD,15,Font.ITALIC));// 创建块定并义字体属性和添加内容
			document.add(chunk1);
			Chunk chunk2 = new Chunk("Text chunk2",FontFactory.getFont(FontFactory.COURIER_BOLD,30,Font.BOLD));// 创建块定并义字体属性和添加内容
			document.add(chunk2);
			document.close();				//关闭文档
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (DocumentException e) {
			e.printStackTrace();
		}
	}
}
